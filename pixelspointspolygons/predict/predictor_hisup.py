# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import functional as F
from shapely.geometry import Polygon

from ..misc import *
from ..models.hisup import HiSupModel

from .predictor import Predictor


class HiSupPredictor(Predictor):

    def predict_with_overlap():
        
        # this code should be in the original HiSup repo in the INRIA predictions
        pass
    
    def setup_model_and_load_checkpoint(self):
        
        self.model = HiSupModel(self.cfg, self.local_rank)
        self.model.eval()
        self.model.to(self.cfg.device)
        self.load_checkpoint()
    
    def predict(self):
        
        self.setup_model_and_load_checkpoint()
        
        loader = self.get_val_loader()
        
        self.logger.info(f"Predicting on {len(loader)} batches...")
        
        loader = self.progress_bar(loader)

        coco_predictions = []
        
        for x_image, x_lidar, y, tile_ids in loader:
            
            batch_size = x_image.size(0) if self.cfg.use_images else x_lidar.size(0)
            
            if self.cfg.use_images:
                x_image = x_image.to(self.cfg.device, non_blocking=True)
            if self.cfg.use_lidar:
                x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)
                
            y=self.to_single_device(y,self.cfg.device)

            polygon_output, loss_dict = self.model(x_image, x_lidar, y)

            ## polygon stuff
            polygon_output = self.to_single_device(polygon_output, 'cpu')
            batch_scores = polygon_output['scores']
            batch_polygons = polygon_output['polys_pred']

            for b in range(batch_size):

                scores = batch_scores[b]
                polys = batch_polygons[b]

                image_result = generate_coco_ann(polys, tile_ids[b], scores=scores)
                if len(image_result) != 0:
                    coco_predictions.extend(image_result)


        return loss_dict, coco_predictions
        
    
    def predict_image(self,infile,outfile=None):
        
        if not os.path.isfile(infile):
            raise FileExistsError(f"Image file {infile} not found.")
        
        if outfile is None:
            outfile = infile.replace(".tif", "_hisup.png")
        
        self.setup_model_and_load_checkpoint()
        
        image_pil = np.array(Image.open(infile).convert("RGB"))

        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1, 1, figsize=(image_pil.shape[0]*px, image_pil.shape[1]*px))
        
        
        with torch.no_grad():
            image = torch.from_numpy(image_pil).permute(2, 0, 1).unsqueeze(0).to(self.cfg.device).to(torch.float32)
            image = F.normalize(image, mean=self.cfg.dataset.image_mean, std=self.cfg.dataset.image_std)

            output, loss = self.model.forward_val(x_images=image, x_lidar=None, y=None)

            if not output['polys_pred']:
                self.logger.warning(f"No polygons predicted for image {infile}.")
                return
            
            polygons = []
            for poly in output['polys_pred'][0]:
                polygons.append(Polygon(poly))
                
            
            plot_image(image_pil, ax=ax)
            plot_shapely_polygons(polygons, ax=ax,pointcolor=[1,1,0],edgecolor=[1,0,1])

            plt.show(block=True)
            
            self.logger.info(f"Save predicted image to {outfile}")
            plt.savefig(outfile)
            
            
