# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.distributed as dist

from functools import partial

from ..predictor import Predictor
from ...models.ffl.local_utils import batch_to_cpu, split_batch, list_of_dicts_to_dict_of_lists, flatten_dict


from . import inference
from . import save_utils
from . import polygonize

class FFLPredictor(Predictor):
    
    def predict(self):
        
        # TODO: just init the model and load checkpoint here and then call predict_from_loader
        # Loading model
        self.setup_model()
        self.load_checkpoint()
        self.init_ddp()
        self.predict_from_loader(self.model, self.loader, save_individual_outputs=self.cfg.model.eval.save_individual_outputs.seg_mask)
        
        raise NotImplementedError("Predict method not implemented in FFLPredictor class.")
    

        
    
    def predict_from_loader(self, model, loader, save_individual_outputs=False, save_aggregated_outputs=True, split_name="val"):
        
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the validation dataset. However, the coco evaluation expects the full validation set, so the its metrics will not be very useful.")
        
        model.eval()
        
            
        tile_data_list = []
        for tile_i, tile_data in enumerate(loader):
            # --- Inference, add result to tile_data_list
            if self.cfg.model.eval.patch_size is not None:
                # Cut image into patches for inference
                inference.inference_with_patching(self.cfg, model, tile_data)
            else:
                # Feed images as-is to the model
                inference.inference_no_patching(self.cfg, model, tile_data)

            tile_data_list.append(tile_data)

            # --- Accumulate batches into tile_data_list until capacity is reached (or this is the last batch)
            if self.cfg.model.batch_size <= len(tile_data_list) or tile_i == len(loader) - 1:
                # Concat tensors of tile_data_list
                accumulated_tile_data = {}
                for key in tile_data_list[0].keys():
                    if isinstance(tile_data_list[0][key], list):
                        accumulated_tile_data[key] = [item for _tile_data in tile_data_list for item in _tile_data[key]]
                    elif isinstance(tile_data_list[0][key], torch.Tensor):
                        accumulated_tile_data[key] = torch.cat([_tile_data[key] for _tile_data in tile_data_list], dim=0)
                    else:
                        print(f"Skipping key {key}")
                        # raise TypeError(f"Type {type(tile_data_list[0][key])} is not handled!")
                tile_data_list = []  # Empty tile_data_list
            else:
                # tile_data_list is not full yet, continue running inference...
                continue

            # --- Polygonize
            crossfield = accumulated_tile_data.get("crossfield", None)
            accumulated_tile_data["polygons"], accumulated_tile_data["polygon_probs"] = polygonize.polygonize(
                self.cfg.model.polygonization, accumulated_tile_data["seg"],
                crossfield_batch=crossfield,
                pool=None)

            # --- Save output
            if self.cfg.model.eval.save_individual_outputs.seg_mask or \
                    self.cfg.model.eval.save_aggregated_outputs.seg_coco:
                # Take seg_interior:
                seg_pred_mask = self.cfg.model.eval.seg_threshold < accumulated_tile_data["seg"][:, 0, ...]
                accumulated_tile_data["seg_mask"] = seg_pred_mask

            accumulated_tile_data = batch_to_cpu(accumulated_tile_data)
            sample_list = split_batch(accumulated_tile_data)

            # # Save individual outputs:
            # if save_individual_outputs:
            #     for sample in sample_list:
            #         # saver_async.add_work(sample)  # TODO: fix bug in using saver_async
            #         save_outputs_partial(sample)

            # # Store aggregated outputs:
            # if save_aggregated_outputs:
            #     self.shared_dict["name_list"].extend(accumulated_tile_data["name"])
            #     if self.cfg.model.eval.save_aggregated_outputs.seg_coco:
            #         for sample in sample_list:
            #             annotations = save_utils.seg_coco(sample)
            #             self.shared_dict["seg_coco_list"].extend(annotations)
            #     if self.cfg.model.eval.save_aggregated_outputs.poly_coco:
            #         for sample in sample_list:
            #             annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"].item())
            #             self.shared_dict["poly_coco_list"].append(annotations)  # annotations could be a dict, or a list
            # # END of loop over samples
            
            
            annotations_list = []
            for sample in sample_list:
                annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"])
                annotations_list.append(annotations)  # annotations could be a dict, or a list
            
            if self.cfg.multi_gpu:
                dist.barrier()
            
            annotations = list_of_dicts_to_dict_of_lists(annotations_list)
            annotations = flatten_dict(annotations)
            return annotations

                
            # # Save aggregated results
            # if save_aggregated_outputs:
            #     self.barrier.wait()  # Wait on all processes so that shared_dict is synchronized.
            #     if self.gpu == 0:
            #         if self.cfg.model.eval.save_aggregated_outputs.stats:
            #             print("Start saving stats:")
            #             # Save sample_stats in CSV:
            #             t1 = time.time()
            #             stats_filepath = os.path.join(self.eval_dirpath, "{}.stats.csv".format(split_name))
            #             stats_file = open(stats_filepath, "w")
            #             fnames = ["name", "iou"]
            #             writer = csv.DictWriter(stats_file, fieldnames=fnames)
            #             writer.writeheader()
            #             for name, iou in sorted(zip(self.shared_dict["name_list"], self.shared_dict["iou_list"]), key=lambda pair: pair[0]):
            #                 writer.writerow({
            #                     "name": name,
            #                     "iou": iou
            #                 })
            #             stats_file.close()
            #             print(f"Finished in {time.time() - t1:02}s")

            #         if self.cfg.model.eval.save_aggregated_outputs.seg_coco:
            #             print("Start saving seg_coco:")
            #             t1 = time.time()
            #             seg_coco_filepath = os.path.join(self.eval_dirpath, "{}.annotation.seg.json".format(split_name))
            #             python_utils.save_json(seg_coco_filepath, list(self.shared_dict["seg_coco_list"]))
            #             print(f"Finished in {time.time() - t1:02}s")

            #         if self.cfg.model.eval.save_aggregated_outputs.poly_coco:
            #             print("Start saving poly_coco:")
            #             poly_coco_base_filepath = os.path.join(self.eval_dirpath, f"{split_name}.annotation.poly")
            #             t1 = time.time()
            #             save_utils.save_poly_coco(self.shared_dict["poly_coco_list"], poly_coco_base_filepath)
            #             print(f"Finished in {time.time() - t1:02}s")



