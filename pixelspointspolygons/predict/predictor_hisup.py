# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import logging
import time
import json
import cv2
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

from tqdm import tqdm

from ..misc import *

class HiSupPredictor:
    def __init__(self, cfg, local_rank=0, world_size=1):
        self.cfg = cfg
        
        self.local_rank = local_rank
        self.world_size = world_size
        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        if verbosity == logging.INFO and local_rank != 0:
            verbosity = logging.WARNING
        self.verbosity = verbosity
        self.update_pbar_every = cfg.update_pbar_every

        self.logger = make_logger(f"Predictor (rank {local_rank})",level=verbosity)
        self.logger.log(logging.INFO, f"Init Predictor on rank {local_rank} in world size {world_size}...")
        self.logger.info(f"Create output directory {cfg.output_dir}")
        if self.local_rank == 0:
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
    
    def load_checkpoint(self, model):
        
        if self.cfg.checkpoint_file is not None:
            checkpoint_file = self.cfg.checkpoint_file
            self.cfg.checkpoint = os.path.basename(checkpoint_file).split(".")[0]+"_overwrite"
        else:
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"{self.cfg.checkpoint}.pth")
        if not os.path.isfile(checkpoint_file):
            raise FileExistsError(f"Checkpoint file {checkpoint_file} not found.")
        self.logger.info(f"Loading model from checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.cfg.device)
        
        # check for correct model type
        cfg = checkpoint.get("cfg")
        if not cfg.use_lidar == self.cfg.use_lidar:
            self.logger.error(f"Model checkpoint was trained with use_lidar={cfg.use_lidar}, but current config is use_lidar={self.cfg.use_lidar}.")
            raise ValueError("Model checkpoint and current config do not match.")
        if not cfg.use_images == self.cfg.use_images:
            self.logger.error(f"Model checkpoint was trained with use_images={cfg.use_images}, but current config is use_images={self.cfg.use_images}.")
            raise ValueError("Model checkpoint and current config do not match.")
        
        if hasattr(cfg, "model.fusion") and isattr(self.cfg.model, "fusion"):
            if not cfg.model.fusion == self.cfg.model.fusion:
                self.logger.error(f"Model checkpoint was trained with fusion={cfg.model.fusion}, but current config is fusion={self.cfg.model.fusion}.")
                raise ValueError("Model checkpoint and current config do not match.")   
        
        
        single_gpu_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
        # single_gpu_state_dict = {k.replace("encoder.image_encoder", "encoder.encoder"): v for k, v in checkpoint["state_dict"].items()}
        
        # single_gpu_state_dict = checkpoint["state_dict"]
        model.load_state_dict(single_gpu_state_dict)
        epoch = checkpoint['epoch']
        self.logger.info(f"Model loaded from epoch: {epoch}")
        
    def predict_with_overlap():
        
        # this could should be in the original HiSup repo in the INRIA predictions
        pass