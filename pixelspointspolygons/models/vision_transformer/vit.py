import timm
import logging
import torch.nn as nn

from ...misc import make_logger

# from timm.models import VisionTransformer

class ViT(nn.Module):
    
    def __init__(self, cfg, local_rank=0) -> None:
        super().__init__()
        self.cfg = cfg
        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)
        
        self.model = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained,
            checkpoint_path=cfg.encoder.checkpoint_file
        )
        self.bottleneck = nn.AdaptiveAvgPool1d(cfg.encoder.out_feature_dim)
    
    def forward(self, x_images):
        
        features = self.model(x_images)
        return self.bottleneck(features[:, 1:,:])