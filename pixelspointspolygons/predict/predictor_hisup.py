# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import time
import torch
import laspy
import json

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from ..misc import *
from ..misc.shared_utils import to_single_device
from ..models.hisup import HiSupModel

from .predictor import Predictor

from ..datasets import get_train_loader, get_val_loader, get_test_loader
from ..misc.debug_visualisations import plot_image, plot_point_cloud, plot_mask, plot_shapely_polygons

class HiSupPredictor(Predictor):

    def predict_with_overlap():
        
        # this code should be in the original HiSup repo in the INRIA predictions
        pass
    
    def setup_model(self):
        
        self.model = HiSupModel(self.cfg, self.local_rank)
        self.model.eval()
        self.model.to(self.cfg.host.device)
    
    def predict_dataset(self, split="val"):
        """This is for predicting the test dataset. Currently just used for debug stuff on val dataset..."""
        
        self.setup_model()
        self.load_checkpoint()

        if split == "train":
            self.loader = get_train_loader(self.cfg,logger=self.logger)
        elif split == "val":
            self.loader = get_val_loader(self.cfg,logger=self.logger)
        elif split == "test":
            self.loader = get_test_loader(self.cfg,logger=self.logger)
        else:   
            raise ValueError(f"Unknown split {split}.")
        
        self.logger.info(f"Predicting on {len(self.loader)} batches...")
        
        loader = self.progress_bar(self.loader)

        coco_predictions = []
        
        with torch.no_grad():
            t0 = time.time()

            for x_image, x_lidar, y, tile_ids in loader:
                
                batch_size = x_image.size(0) if self.cfg.experiment.encoder.use_images else x_lidar.size(0)
                
                if self.cfg.experiment.encoder.use_images:
                    x_image = x_image.to(self.cfg.host.device, non_blocking=True)
                if self.cfg.experiment.encoder.use_lidar:
                    x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)
                    
                y=to_single_device(y,self.cfg.host.device)

                polygon_output, loss_dict = self.model(x_image, x_lidar, y)

                ## polygon stuff
                polygon_output = to_single_device(polygon_output, 'cpu')
                batch_scores = polygon_output['scores']
                batch_polygons = polygon_output['polys_pred']

                for b in range(batch_size):

                    scores = batch_scores[b]
                    polys = batch_polygons[b]

                    image_result = generate_coco_ann(polys, tile_ids[b], scores=scores)
                    if len(image_result) != 0:
                        coco_predictions.extend(image_result)
        
            self.logger.info(f"Average prediction speed: {(time.time() - t0) / len(self.loader.dataset):.2f} [s / image]")
            time_dict = {}
            time_dict["prediction_time"] = (time.time() - t0) / len(self.loader.dataset)
        
            if self.local_rank == 0:
                if not len(coco_predictions):
                    self.logger.warning("No polygons predicted. Check your model and data loader.")
                else:
                    self.logger.info(f"Predicted {len(coco_predictions)} polygons.")
                os.makedirs(os.path.dirname(self.cfg.evaluation.pred_file), exist_ok=True)
                self.logger.info(f"Writing predictions to {self.cfg.evaluation.pred_file}")
                with open(self.cfg.evaluation.pred_file, "w") as fp:
                    fp.write(json.dumps(coco_predictions))
            
            return time_dict

    
    def predict_file(self,img_infile=None,lidar_infile=None,outfile=None):
        

        image, image_pillow = self.load_image_from_file(img_infile)
        lidar = self.load_lidar_from_file(lidar_infile)
        
        self.setup_model()
        self.load_checkpoint()
        
        with torch.no_grad():

            output, loss = self.model.forward_val(x_images=image, x_lidar=lidar, y=None)

            self.plot_prediction(output['polys_pred'][0], image=image, image_np=image_pillow, lidar=lidar, outfile=outfile)

            
            
