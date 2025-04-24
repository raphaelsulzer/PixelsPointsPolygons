import logging
import torch.nn as nn

from ..multitask_head import MultitaskHead
from .pointpillars_vit import PointPillarsViT

from ...misc.logger import make_logger

class PointPillarsViTCNN(nn.Module):
    
    def __init__(self, cfg, local_rank=0) -> None:
        super().__init__()
        self.cfg = cfg
        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        self.pp_vit = PointPillarsViT(cfg, local_rank=local_rank)
                                
        self.proj = nn.Sequential(
            nn.Upsample(size=self.cfg.experiment.encoder.out_feature_size, mode='bilinear', align_corners=False),
            nn.Conv2d(self.cfg.experiment.encoder.patch_feature_dim, self.cfg.experiment.model.decoder.in_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.experiment.model.decoder.in_feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, points):
        
        x = self.pp_vit(points)
        
        # x = x[:, 1:,:] # drop CLS token
        
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.proj(x)
        
        return x
