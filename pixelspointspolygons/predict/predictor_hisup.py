# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import laspy
import json

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import functional as F
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler

from ..misc import *
from ..misc.shared_utils import to_single_device
from ..models.hisup import HiSupModel

from .predictor import Predictor

from ..datasets import get_train_loader, get_val_loader, get_test_loader


class HiSupPredictor(Predictor):

    def predict_with_overlap():
        
        # this code should be in the original HiSup repo in the INRIA predictions
        pass
    
    def setup_model_and_load_checkpoint(self):
        
        self.model = HiSupModel(self.cfg, self.local_rank)
        self.model.eval()
        self.model.to(self.cfg.device)
        self.load_checkpoint()
    
    def predict_dataset(self, split="val"):
        """This is for predicting the test dataset. Currently just used for debug stuff on val dataset..."""
        
        self.setup_model_and_load_checkpoint()
        
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
        
            for x_image, x_lidar, y, tile_ids in loader:
                
                batch_size = x_image.size(0) if self.cfg.use_images else x_lidar.size(0)
                
                if self.cfg.use_images:
                    x_image = x_image.to(self.cfg.device, non_blocking=True)
                if self.cfg.use_lidar:
                    x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)
                    
                y=to_single_device(y,self.cfg.device)

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


            os.makedirs(os.path.dirname(self.cfg.eval.pred_file), exist_ok=True)
            self.logger.info(f"Writing predictions to {self.cfg.eval.pred_file}")
            with open(self.cfg.eval.pred_file, "w") as fp:
                fp.write(json.dumps(coco_predictions))
        
    
    def predict_file(self,img_infile=None,lidar_infile=None,outfile=None):
        
        if img_infile is not None and not os.path.isfile(img_infile):
            raise FileExistsError(f"Image file {img_infile} not found.")
        
        if lidar_infile is not None and not os.path.isfile(lidar_infile):
            raise FileExistsError(f"Image file {lidar_infile} not found.")
        
        if img_infile is None and lidar_infile is None:
            raise ValueError("Either image or lidar file must be provided.")
        
        if img_infile is not None:
            image_pil = np.array(Image.open(img_infile).convert("RGB"))
            image = torch.from_numpy(image_pil).permute(2, 0, 1).unsqueeze(0).to(self.cfg.device).to(torch.float32)
            image = F.normalize(image, mean=self.cfg.dataset.image_mean, std=self.cfg.dataset.image_std)
        else:
            image = None

        if lidar_infile is not None:
            las = laspy.read(lidar_infile)
            lidar = np.vstack((las.x, las.y, las.z)).transpose()
            
            img_dim = 512
            img_res = 0.25

            lidar[:, :2] = (lidar[:, :2] - np.min(lidar,axis=0)[:2]) / img_res
            lidar[:, 1] = img_dim - lidar[:, 1]

            # # scale z vals to [0,100]
            scaler = MinMaxScaler(feature_range=(0,512))
            lidar[:, -1] = scaler.fit_transform(lidar[:, -1].reshape(-1, 1)).squeeze()
            
            lidar = torch.from_numpy(lidar).unsqueeze(0).to(self.cfg.device).to(torch.float32).contiguous()
        
        
        
            
            
        else:
            lidar = None
        
        if outfile is None:
            outfile = "prediction.png"
        
        self.setup_model_and_load_checkpoint()
        

        px = 1/plt.rcParams['figure.dpi']
        # fig, ax = plt.subplots(1, 1, figsize=(image_pil.shape[0]*px, image_pil.shape[1]*px))
        fig, ax = plt.subplots(1, 1, figsize=(512*px, 512*px))
        
        
        with torch.no_grad():

            output, loss = self.model.forward_val(x_images=image, x_lidar=lidar, y=None)

            if not len(output['polys_pred'][0]):
                self.logger.warning(f"No polygons predicted.")
                return
            
            polygons = []
            for poly in output['polys_pred'][0]:
                polygons.append(Polygon(poly))
                
            if img_infile is not None:
                plot_image(image_pil, ax=ax)
            
            if lidar_infile is not None:
                plot_point_cloud(lidar, ax=ax)
                
            plot_mask(output["mask_pred"][0]>0.5, color=[1,0,1,0.5], ax=ax)
            
            plot_shapely_polygons(polygons, ax=ax,pointcolor=[1,1,0],edgecolor=[1,0,1])
            
            self.logger.info(f"Save prediction to {outfile}")
            plt.savefig(outfile)
            
            
