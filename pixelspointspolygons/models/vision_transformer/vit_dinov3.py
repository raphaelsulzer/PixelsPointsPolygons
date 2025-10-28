import logging
import os

import torch
import torch.nn as nn

from ...misc import make_logger

class ViTDINOv3(nn.Module):
    
    def __init__(self, cfg, bottleneck=False, local_rank=0) -> None:
        super().__init__()
        self.cfg = cfg
        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)
        
        
        # Safety check
        checkpoint_path = cfg.experiment.encoder.checkpoint_file
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
        
        # Step 1: Load model architecture
        self.vit = torch.hub.load(self.cfg.host.dino_v3_repo, self.cfg.experiment.encoder.type, pretrained=False, source='local')
        self.norm = self.vit.norm

        # Step 2: Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=self.cfg.host.device)

        # If checkpoint has extra keys (like 'model' or 'state_dict'), unwrap it
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Step 3: Load state dict into the model
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        self.logger.debug(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.debug(f"Missing keys: {missing}")
        self.logger.debug(f"Unexpected keys: {unexpected}")
        

        # Optional bottleneck (e.g. to reduce token dimension)
        if bottleneck:
            self.bottleneck = nn.AdaptiveAvgPool1d(cfg.experiment.encoder.out_feature_dim)
        else:
            self.bottleneck = nn.Identity()

    def forward(self, x):
        # Extract only patch_token without CLS token
        patch_tokens = self.vit.forward_features(x)["x_norm_patchtokens"]
        patch_tokens = self.bottleneck(patch_tokens)
        return patch_tokens