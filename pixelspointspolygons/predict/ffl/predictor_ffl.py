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

from collections import defaultdict
from multiprocessing import Pool

from ...models.ffl.local_utils import batch_to_cpu, split_batch, list_of_dicts_to_dict_of_lists, flatten_dict
from ...models.ffl.model_ffl import FFLModel
from ...datasets import get_val_loader

from ..predictor import Predictor

from . import inference
from . import save_utils
from . import polygonize

class FFLPredictor(Predictor):
    
    def predict_dataset(self):
        
        self.logger.info(f"Starting prediction and polygonization...")

        # Loading model
        self.model = FFLModel(self.cfg, self.local_rank)
        self.load_checkpoint()
        
        self.loader = get_val_loader(self.cfg,logger=self.logger)
        annotations = self.predict_from_loader(self.model, self.loader)
        
        for k,coco_predictions in annotations.items():
            outfile = os.path.join(os.path.dirname(self.cfg.eval.pred_file), k, f"{self.cfg.checkpoint}.json")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.logger.info(f"Saving prediction {k} to {outfile}")
            with open(outfile, "w") as fp:
                fp.write(json.dumps(coco_predictions))
        
        self.logger.info(f"Copy asm.tol_1 to predictions/{self.cfg.checkpoint}.json")
        if "asm.tol_1" in annotations.keys():
            outfile = os.path.join(os.path.dirname(self.cfg.eval.pred_file), f"{self.cfg.checkpoint}.json")
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, "w") as fp:
                fp.write(json.dumps(annotations["asm.tol_1"]))
        
        
    def predict_from_loader(self, model, loader):
        
        self.logger.debug(f"Prediction from {self.cfg.checkpoint}")
        self.logger.debug(f"Polygonization with method {self.cfg.polygonization.method}")
        
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the validation dataset. However, the coco evaluation expects the full validation set, so the its metrics will not be very useful.")
        
        model.eval()
            
        annotations_list = []
        
        for batch in self.progress_bar(loader):
            
            batch_size = batch["image"].shape[0] if self.cfg.use_images else batch["lidar"].shape[0]
                        
            # --- Inference, add result to batch_list
            if self.cfg.model.eval.patch_size is not None:
                # Cut image into patches for inference
                batch = inference.inference_with_patching(self.cfg, model, batch)
                pool = None
            else:
                # Feed images as-is to the model
                batch = inference.inference_no_patching(self.cfg, model, batch)
                num_workers = self.cfg.run_type.num_workers
                # pool = Pool(processes=num_workers) if num_workers > 0 else None
                pool = None # there is some skan error when I try with Pool()
            
            #     # --- Polygonize
            try:
                batch["polygons"], batch["polygon_probs"] = polygonize.polygonize(
                    self.cfg.polygonization, batch["seg"],
                    crossfield_batch=batch.get("crossfield", None),
                    pool=pool)
            except Exception as e:
                batch = batch_to_cpu(batch)
                # raise e
                self.logger.error(f"Polygonization failed: {e}")
                self.logger.error("Skipping this batch...")
                continue

            batch = batch_to_cpu(batch)
            sample_list = split_batch(batch,batch_size=batch_size)
            
            for sample in sample_list:
                annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"])
                annotations_list.append(annotations)  # annotations could be a dict, or a list
                    
        # else:
        #     self.logger.info(f"Rank {self.local_rank} waiting until polygonization is done...")
        if self.cfg.multi_gpu:
            # dist.barrier()
                        
            # Gather the list of dictionaries from all ranks
            temp = [None] * self.world_size  # Placeholder for gathered objects
            dist.all_gather_object(temp, annotations_list)

            # Flatten the list of lists into a single list
            annotations_list = [item for sublist in temp for item in sublist]

        if len(annotations_list):
            annotations = list_of_dicts_to_dict_of_lists(annotations_list)
            annotations = flatten_dict(annotations)
            return annotations
        else:
            return dict()