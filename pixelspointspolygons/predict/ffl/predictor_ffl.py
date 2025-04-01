# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from . import inference

class FFLPredictor:
    
    def predict(self):
        
        # TODO: take the rest of the code from FFL evaluator.evaluate() which is not already included in predict_from_loader()
        
        raise NotImplementedError("Predict method not implemented in FFLPredictor class.")
        
    
    def predict_from_loader(self, model, loader):
        
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the validation dataset. However, the coco evaluation expects the full validation set, so the its metrics will not be very useful.")
        
        model.eval()
        
        coco_predictions = []
        for tile_i, tile_data in enumerate(loader):
            # --- Inference, add result to tile_data_list
            if self.config["eval_params"]["patch_size"] is not None:
                # Cut image into patches for inference
                inference.inference_with_patching(self.config, self.model, tile_data)
            else:
                # Feed images as-is to the model
                inference.inference_no_patching(self.config, self.model, tile_data)

            tile_data_list.append(tile_data)

            # --- Accumulate batches into tile_data_list until capacity is reached (or this is the last batch)
            if self.config["eval_params"]["batch_size_mult"] <= len(tile_data_list)\
                    or tile_i == len(tile_iterator) - 1:
                # Concat tensors of tile_data_list
                accumulated_tile_data = {}
                for key in tile_data_list[0].keys():
                    if isinstance(tile_data_list[0][key], list):
                        accumulated_tile_data[key] = [item for _tile_data in tile_data_list for item in _tile_data[key]]
                    elif isinstance(tile_data_list[0][key], torch.Tensor):
                        accumulated_tile_data[key] = torch.cat([_tile_data[key] for _tile_data in tile_data_list], dim=0)
                    else:
                        raise TypeError(f"Type {type(tile_data_list[0][key])} is not handled!")
                tile_data_list = []  # Empty tile_data_list
            else:
                # tile_data_list is not full yet, continue running inference...
                continue

            # --- Polygonize
            crossfield = accumulated_tile_data["crossfield"] if "crossfield" in accumulated_tile_data else None
            accumulated_tile_data["polygons"], accumulated_tile_data["polygon_probs"] = polygonize.polygonize(
                self.config["polygonize_params"], accumulated_tile_data["seg"],
                crossfield_batch=crossfield,
                pool=pool)

            # --- Save output
            if self.config["eval_params"]["save_individual_outputs"]["seg_mask"] or \
                    self.config["eval_params"]["save_aggregated_outputs"]["seg_coco"]:
                # Take seg_interior:
                seg_pred_mask = self.config["eval_params"]["seg_threshold"] < accumulated_tile_data["seg"][:, 0, ...]
                accumulated_tile_data["seg_mask"] = seg_pred_mask

            accumulated_tile_data = local_utils.batch_to_cpu(accumulated_tile_data)
            sample_list = local_utils.split_batch(accumulated_tile_data)

            # Save individual outputs:
            if save_individual_outputs:
                for sample in sample_list:
                    # saver_async.add_work(sample)  # TODO: fix bug in using saver_async
                    save_outputs_partial(sample)

            # Store aggregated outputs:
            if save_aggregated_outputs:
                self.shared_dict["name_list"].extend(accumulated_tile_data["name"])
                if self.config["eval_params"]["save_aggregated_outputs"]["stats"]:
                    y_pred = accumulated_tile_data["seg"][:, 0, ...].cpu()
                    if "gt_mask" in accumulated_tile_data:
                        y_true = accumulated_tile_data["gt_mask"][:, 0, ...]
                    elif "gt_polygons_image" in accumulated_tile_data:
                        y_true = accumulated_tile_data["gt_polygons_image"][:, 0, ...]
                    else:
                        raise ValueError("Either gt_mask or gt_polygons_image should be in accumulated_tile_data")
                    iou = measures.iou(y_pred.reshape(y_pred.shape[0], -1), y_true.reshape(y_true.shape[0], -1),
                                       threshold=self.config["eval_params"]["seg_threshold"])
                    self.shared_dict["iou_list"].extend(iou.cpu().numpy())
                if self.config["eval_params"]["save_aggregated_outputs"]["seg_coco"]:
                    for sample in sample_list:
                        annotations = save_utils.seg_coco(sample)
                        self.shared_dict["seg_coco_list"].extend(annotations)
                if self.config["eval_params"]["save_aggregated_outputs"]["poly_coco"]:
                    for sample in sample_list:
                        annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"].item())
                        self.shared_dict["poly_coco_list"].append(annotations)  # annotations could be a dict, or a list