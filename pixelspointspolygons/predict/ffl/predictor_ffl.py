# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import json
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.distributed as dist

from ...models.ffl.local_utils import batch_to_cpu, split_batch, list_of_dicts_to_dict_of_lists, flatten_dict
from ...models.ffl.model_ffl import FFLModel
from ...datasets import get_train_loader, get_val_loader

from ..predictor import Predictor

from . import inference
from . import save_utils
from . import polygonize

class FFLPredictor(Predictor):
    
    def predict(self):
        
        self.logger.info(f"Starting prediction and polygonization...")

        # Loading model
        self.model = FFLModel(self.cfg, self.local_rank)
        self.load_checkpoint(self.model)
        
        self.loader = get_val_loader(self.cfg,logger=self.logger)
        annotations = self.predict_from_loader(self.model, self.loader)
        
        for k,coco_predictions in annotations.items():
            outfile = os.path.join(os.path.dirname(self.cfg.eval.pred_file), k, f"{self.cfg.checkpoint}.json")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.logger.info(f"Saving prediction {k} to {outfile}")
            with open(outfile, "w") as fp:
                fp.write(json.dumps(coco_predictions))

        
    def predict_from_loader(self, model, loader):
        
        self.logger.debug(f"Prediction from {self.cfg.checkpoint}")
        self.logger.debug(f"Polygonization with method {self.cfg.model.polygonization.method}")
        
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the validation dataset. However, the coco evaluation expects the full validation set, so the its metrics will not be very useful.")
        
        model.eval()
            
        tile_data_list = []
        annotations_list = []
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
            if self.cfg.model.batch_size <= len(tile_data_list) or (tile_i == len(loader) - 1):
                # Concat tensors of tile_data_list
                accumulated_tile_data = {}
                for key in tile_data_list[0].keys():
                    if isinstance(tile_data_list[0][key], list):
                        accumulated_tile_data[key] = [item for _tile_data in tile_data_list for item in _tile_data[key]]
                    elif isinstance(tile_data_list[0][key], torch.Tensor):
                        accumulated_tile_data[key] = torch.cat([_tile_data[key] for _tile_data in tile_data_list], dim=0)
                    else:
                        pass
                        # print(f"Skipping key {key}")
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
            
            for sample in sample_list:
                annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"])
                annotations_list.append(annotations)  # annotations could be a dict, or a list
             
        annotations = list_of_dicts_to_dict_of_lists(annotations_list)
        annotations = flatten_dict(annotations)
        return annotations