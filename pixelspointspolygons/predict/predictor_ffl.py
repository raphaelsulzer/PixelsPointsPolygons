# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import time
import json
import torch
import laspy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import functional as F
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler

from ..models.ffl.local_utils import batch_to_cpu, split_batch, list_of_dicts_to_dict_of_lists, flatten_dict
from ..models.ffl.model_ffl import FFLModel
from ..datasets import get_train_loader, get_val_loader, get_test_loader
from ..misc.debug_visualisations import plot_image, plot_point_cloud, plot_mask, plot_shapely_polygons

from .predictor import Predictor

from .ffl import inference
from .ffl import save_utils
from .ffl import polygonize

class FFLPredictor(Predictor):
    
    def setup_model_and_load_checkpoint(self):
        
        self.model = FFLModel(self.cfg, self.local_rank)
        self.model.eval()
        self.model.to(self.cfg.host.device)
        self.load_checkpoint()
    
    def predict_dataset(self, split="val"):
        
        self.logger.info(f"Starting prediction and polygonization...")

        # Loading model
        self.setup_model_and_load_checkpoint()
        
        if split == "train":
            self.loader = get_train_loader(self.cfg,logger=self.logger)
        elif split == "val":
            self.loader = get_val_loader(self.cfg,logger=self.logger)
        elif split == "test":
            self.loader = get_test_loader(self.cfg,logger=self.logger)
        else:   
            raise ValueError(f"Unknown split {split}.")
        
        t0 = time.time()
        annotations = self.predict_from_loader(self.model, self.loader)
        self.logger.info(f"Average prediction speed: {(time.time() - t0) / len(self.loader.dataset):.2f} [s / image]")
        time_dict = {}
        time_dict["prediction_time"] = (time.time() - t0) / len(self.loader.dataset)
        
        if self.local_rank == 0:

            for k,coco_predictions in annotations.items():
                outfile = os.path.join(os.path.dirname(self.cfg.evaluation.pred_file), k, f"{self.cfg.checkpoint}.json")
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                self.logger.info(f"Saving prediction {k} to {outfile}")
                with open(outfile, "w") as fp:
                    fp.write(json.dumps(coco_predictions))
            
            self.logger.info(f"Copy acm.tol_1 to predictions_{split}/{self.cfg.checkpoint}.json")
            if "acm.tol_1" in annotations.keys():
                outfile = os.path.join(os.path.dirname(self.cfg.evaluation.pred_file), f"{self.cfg.checkpoint}.json")
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                with open(outfile, "w") as fp:
                    fp.write(json.dumps(annotations["acm.tol_1"]))
        
        
        return time_dict
        
    def predict_from_loader(self, model, loader):
        
        self.logger.debug(f"Prediction from {self.cfg.checkpoint}")
        self.logger.debug(f"Polygonization with method {self.cfg.experiment.polygonization.method}")
        
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the validation dataset. However, the coco evaluation expects the full validation set, so the its metrics will not be very useful.")
        
        model.eval()
            
        annotations_list = []
        
        for batch in self.progress_bar(loader):
            
            batch_size = batch["image"].shape[0] if self.cfg.experiment.encoder.use_images else batch["lidar"].shape[0]
                        
            # --- Inference, add result to batch_list
            if self.cfg.experiment.model.eval.patch_size is not None:
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
                    self.cfg.experiment.polygonization, batch["seg"],
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
        if self.cfg.host.multi_gpu:
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
        
        
        
    def predict_file(self,img_infile=None,lidar_infile=None,outfile=None):
        
        
        image, image_pillow = self.load_image_from_file(img_infile)
        lidar = self.load_lidar_from_file(lidar_infile)
        
        self.setup_model_and_load_checkpoint()
        
            
        batch = {}
        if image is not None:
            batch["image"] = image
        if lidar is not None:
            batch["lidar"] = lidar
        
        batch = inference.inference_no_patching(self.cfg, self.model, batch)
        batch["polygons"], batch["polygon_probs"] = polygonize.polygonize(
            self.cfg.experiment.polygonization, batch["seg"],
            crossfield_batch=batch.get("crossfield", None),
            pool=None)
        batch = batch_to_cpu(batch)
        sample_list = split_batch(batch,batch_size=1)
        
        self.plot_prediction(sample_list[0]["polygons"]['acm']['tol_1'], image=image, image_pillow=image_pillow, lidar=lidar, outfile=outfile)

            
            
