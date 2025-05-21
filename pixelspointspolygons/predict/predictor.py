# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import laspy
import logging
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from torchvision.transforms import functional as F
from shapely.geometry import Polygon

from tqdm import tqdm

from ..misc import *

class Predictor:
    def __init__(self, cfg, local_rank=0, world_size=1):
        self.cfg = cfg
        
        self.local_rank = local_rank
        self.world_size = world_size

        self.device = torch.device(f"cuda:{local_rank}")

        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        self.verbosity = verbosity
        self.update_pbar_every = cfg.host.update_pbar_every

        self.logger.log(logging.INFO, f"Init Predictor on rank {local_rank} in world size {world_size}...")
        if self.local_rank == 0 and not os.path.exists(cfg.output_dir):
            self.logger.info(f"Create output directory {cfg.output_dir}")
            os.makedirs(cfg.output_dir, exist_ok=True)
                           
    def progress_bar(self,item):
        
        disable = self.verbosity >= logging.WARNING
        
        pbar = tqdm(item, total=len(item), 
                      file=sys.stdout, 
                    #   dynamic_ncols=True, 
                      mininterval=self.update_pbar_every,                      
                      disable=disable,
                      position=0,
                      leave=True)
    
        return pbar
    
    def load_checkpoint(self):
        
        ## get the file
        checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"{self.cfg.checkpoint}.pth")
        
        if not os.path.isfile(checkpoint_file):
            raise FileExistsError(f"Checkpoint file {checkpoint_file} not found.")
        self.logger.info(f"Loading model from checkpoint: {checkpoint_file}")
        
        ## load the checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.cfg.host.device)
        for k in checkpoint.keys():
            if "_state_dict" in k:
                checkpoint[k.replace("_state_dict","")] = checkpoint.pop(k)
        
        ## check for correct model type
        cfg = checkpoint.get("cfg",None)
        if cfg is not None:
            if hasattr(cfg, "use_lidar"):
                if not cfg.use_lidar == self.cfg.experiment.encoder.use_lidar:
                    self.logger.error(f"Model checkpoint was trained with use_lidar={cfg.experiment.encoder.use_lidar}, but current config is use_lidar={self.cfg.experiment.encoder.use_lidar}.")
                    raise ValueError("Model checkpoint and current config do not match.")
            if hasattr(cfg, "use_images"):
                if not cfg.use_images == self.cfg.experiment.encoder.use_images:
                    self.logger.error(f"Model checkpoint was trained with use_images={cfg.experiment.encoder.use_images}, but current config is use_images={self.cfg.experiment.encoder.use_images}.")
                    raise ValueError("Model checkpoint and current config do not match.")
            
            if hasattr(cfg, "model.fusion") and isattr(self.cfg.experiment.model, "fusion"):
                if not cfg.experiment.model.fusion == self.cfg.experiment.model.fusion:
                    self.logger.error(f"Model checkpoint was trained with fusion={cfg.experiment.model.fusion}, but current config is fusion={self.cfg.experiment.model.fusion}.")
                    raise ValueError("Model checkpoint and current config do not match.")   
        
        # self.model.load_state_dict(model_state_dict)
        self.model = smart_load_state_dict(self.model, checkpoint["model"], self.logger, strict=True)
        epoch = checkpoint.get("epochs_run",checkpoint.get("epoch",0))
        
        self.logger.info(f"Model loaded from epoch: {epoch}")
        
    def load_image_from_file(self, img_infile):
        
        if img_infile is not None:
            image_pil = np.array(Image.open(img_infile).convert("RGB"))
            image = torch.from_numpy(image_pil).permute(2, 0, 1).unsqueeze(0).to(self.cfg.host.device).to(torch.float32)/255.0
            image = F.normalize(image, mean=self.cfg.experiment.encoder.image_mean, std=self.cfg.experiment.encoder.image_std)
            return image, image_pil
        else:
            return None, None
        
    
    
    def load_lidar_from_file(self, lidar_infile):
        
        img_res = 0.25
        img_dim = 224
        
        if lidar_infile is not None:
            las = laspy.read(lidar_infile)
            lidar = np.vstack((las.x, las.y, las.z)).transpose()
            

            lidar[:, :2] = (lidar[:, :2] - np.min(lidar,axis=0)[:2]) / img_res
            lidar[:, 1] = img_dim - lidar[:, 1]

            # # scale z vals to [0,100]
            scaler = MinMaxScaler(feature_range=(0,self.cfg.experiment.encoder.in_voxel_size.z))
            lidar[:, -1] = scaler.fit_transform(lidar[:, -1].reshape(-1, 1)).squeeze()
            
            lidar = torch.from_numpy(lidar).unsqueeze(0).to(self.cfg.host.device).to(torch.float32).contiguous()
            
            return lidar
        else:
            return None
    
    
    def plot_prediction(self, polygons, image=None, image_pillow=None, lidar=None, outfile=None):
        
        if not len(polygons):
            self.logger.warning(f"No polygons predicted.")
            return
        
        
        if outfile is None:
            if image is not None:
                name = "image"
            if lidar is not None:
                name = "lidar"
            if image is not None and lidar is not None:
                name = "fusion"
            outfile = f"prediction_{self.cfg.experiment.model.name}_{name}.png"
            
        
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1, 1, figsize=(1000*px, 1000*px))
        
        
        shapely_polygons = []
        for poly in polygons:
            if isinstance(poly, Polygon):
                shapely_polygons.append(poly)
            else:
                shapely_polygons.append(Polygon(poly))
        
        alpha = 1.0
        if image is not None:
            alpha = 0.7
            plot_image(image_pillow, ax=ax)
        
        if lidar is not None:
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            plot_point_cloud(lidar, ax=ax, alpha=alpha, pointsize=0.5)
            
        
        plot_shapely_polygons(shapely_polygons, ax=ax,pointcolor=[1,1,0],edgecolor=[1,0,1],linewidth=4,pointsize=8)
        
        self.logger.info(f"Save prediction to {outfile}")
        plt.savefig(outfile)