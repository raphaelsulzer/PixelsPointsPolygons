import timm
import logging
import torch.nn as nn

from ...misc import make_logger

# from timm.models import VisionTransformer

class ViT(nn.Module):
    
    def __init__(self, cfg, bottleneck=False, local_rank=0) -> None:
        super().__init__()
        self.cfg = cfg
        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)
    
        logging.getLogger('timm').setLevel(logging.WARNING)
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.experiment.encoder.pretrained,
            checkpoint_path=cfg.experiment.encoder.checkpoint_file
        )
        
        if bottleneck:
            self.bottleneck = nn.AdaptiveAvgPool1d(cfg.experiment.encoder.out_feature_dim)
        else:
            self.bottleneck = nn.Identity()
            
    
    def forward(self, x):
        
        x = self.vit(x)
        x = self.bottleneck(x[:, 1:,:])
        return x