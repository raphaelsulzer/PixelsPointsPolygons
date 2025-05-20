# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import logging
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

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
    
    def load_checkpoint(self):
        
        ## get the file
        if self.cfg.checkpoint_file is not None:
            checkpoint_file = self.cfg.checkpoint_file
            self.cfg.checkpoint = os.path.basename(checkpoint_file).split(".")[0]+"_overwrite"
        else:
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
            if not cfg.experiment.encoder.use_lidar == self.cfg.experiment.encoder.use_lidar:
                self.logger.error(f"Model checkpoint was trained with use_lidar={cfg.experiment.encoder.use_lidar}, but current config is use_lidar={self.cfg.experiment.encoder.use_lidar}.")
                raise ValueError("Model checkpoint and current config do not match.")
            if not cfg.experiment.encoder.use_images == self.cfg.experiment.encoder.use_images:
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
        
        