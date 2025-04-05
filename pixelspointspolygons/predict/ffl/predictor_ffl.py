# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import json
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from collections import defaultdict

from ...models.ffl.local_utils import batch_to_cpu, split_batch, list_of_dicts_to_dict_of_lists, flatten_dict
from ...models.ffl.model_ffl import FFLModel
from ...datasets import get_val_loader

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
            
        annotations_list = []
        
        for batch in self.progress_bar(loader):
            
            batch_size = batch["image"].shape[0] if self.cfg.use_images else batch["lidar"].shape[0]
                        
            # --- Inference, add result to batch_list
            if self.cfg.model.eval.patch_size is not None:
                # Cut image into patches for inference
                inference.inference_with_patching(self.cfg, model, batch)
            else:
                # Feed images as-is to the model
                inference.inference_no_patching(self.cfg, model, batch)

            # batch_list.append(batch)
            #### this whole thing is totally useless. if you want to have a different batch size for the polygonization, just specify it in the loader!!
            # # --- Accumulate batches into batch_list until capacity is reached (or this is the last batch)
            # if self.cfg.model.batch_size <= len(batch_list) or (tile_i == len(loader) - 1):
            #     # Concat tensors of batch_list
            #     accumulated_batch= {}
            #     for key in batch_list[0].keys():
            #         if isinstance(batch_list[0][key], list):
            #             accumulated_batch[key] = [item for _batchin batch_list for item in _batch[key]]
            #         elif isinstance(batch_list[0][key], torch.Tensor):
            #             accumulated_batch[key] = torch.cat([_batch[key] for _batchin batch_list], dim=0)
            #         else:
            #             pass
            #             # print(f"Skipping key {key}")
            #             # raise TypeError(f"Type {type(batch_list[0][key])} is not handled!")
            #     batch_list = []  # Empty batch_list
            # else:
            #     # batch_list is not full yet, continue running inference...
            #     continue

            # --- Polygonize
            crossfield = batch.get("crossfield", None)
            try:
                batch["polygons"], batch["polygon_probs"] = polygonize.polygonize(
                    self.cfg.model.polygonization, batch["seg"],
                    crossfield_batch=crossfield,
                    pool=None)
            except Exception as e:
                raise e
                self.logger.error(f"Polygonization failed: {e}")
                self.logger.error("Skipping this batch...")
                continue

            batch = batch_to_cpu(batch)
            sample_list = split_batch(batch,batch_size=batch_size)
            
            for sample in sample_list:
                annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"])
                annotations_list.append(annotations)  # annotations could be a dict, or a list
        
        if len(annotations_list):
            annotations = list_of_dicts_to_dict_of_lists(annotations_list)
            annotations = flatten_dict(annotations)
            return annotations
        else:
            return defaultdict(list)