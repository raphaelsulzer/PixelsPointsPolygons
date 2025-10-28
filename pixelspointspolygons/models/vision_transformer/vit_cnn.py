import timm
import logging
import os
import torch

import torch.nn as nn

from ...misc.logger import make_logger
from ...misc import suppress_stdout

class ViTCNN(nn.Module):
    
    def __init__(self, cfg, local_rank=0) -> None:
        super().__init__()
        self.cfg = cfg
        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        if not os.path.isfile(cfg.experiment.encoder.checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file {cfg.experiment.encoder.checkpoint_file} not found.")
        logging.getLogger('timm').setLevel(logging.WARNING)
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.type,
            num_classes=0,
            global_pool='',
            # pretrained=cfg.experiment.encoder.pretrained,
            # checkpoint_path=cfg.experiment.encoder.checkpoint_file
        )
        
        if cfg.experiment.encoder.vit.pretrained:
            self.vit.load_state_dict(torch.load(cfg.experiment.encoder.vit.checkpoint_file, map_location=self.cfg.host.device), strict=False)
    
        self.proj = nn.Sequential(
            nn.Upsample(size=self.cfg.experiment.encoder.out_feature_size, mode='bilinear', align_corners=False),
            nn.Conv2d(self.cfg.experiment.encoder.patch_feature_dim, self.cfg.experiment.model.decoder.in_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.experiment.model.decoder.in_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        ## heads are now defined in the decoder as they should be
        # self.head = MultitaskHead(self.cfg.experiment.model.decoder.in_feature_dim, 2, head_size=[[2]])


    def forward(self, image):
        
        x = self.vit(image)
        
        x = x[:, 1:,:] # drop CLS token
        
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        x = self.proj(x)
        
        return x
