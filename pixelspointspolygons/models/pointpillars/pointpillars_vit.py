import torch
import logging
import timm
import os

import torch.nn as nn

# from .pointpillars_ori import PointPillarsEncoder
from .pointpillars_o3d import PointPillarsEncoder

from ...misc.logger import make_logger

class PointPillarsViT(torch.nn.Module):
    
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "PointPillars".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        scatter: Config of PointPillarsScatter module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self, cfg, bottleneck=False, local_rank=0):
        super().__init__()
        
        self.cfg = cfg        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        if not os.path.isfile(cfg.experiment.encoder.vit.checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file {cfg.experiment.encoder.vit.checkpoint_file} not found.")
        
        logging.getLogger('timm').setLevel(logging.WARNING)
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.vit.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.experiment.encoder.vit.pretrained,
            checkpoint_path=cfg.experiment.encoder.vit.checkpoint_file
        )
        
        # if cfg.experiment.encoder.vit.pretrained:
        #     self.vit.load_state_dict(torch.load(cfg.experiment.encoder.vit.checkpoint_file, map_location=self.cfg.host.device), strict=False)
        
        #### replace VisionTransformer patch embedding with LiDAR encoder        
        output_shape = [cfg.experiment.encoder.patch_feature_width, cfg.experiment.encoder.patch_feature_height]
        voxel_encoder={
            'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
            'feat_channels': [64,cfg.experiment.encoder.patch_feature_dim],
        }
        scatter={
            "in_channels" : cfg.experiment.encoder.patch_feature_dim, 
            "output_shape" : output_shape
        }
        self.vit.patch_embed = PointPillarsEncoder(cfg, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        
        if bottleneck:
            self.bottleneck = nn.AdaptiveAvgPool1d(cfg.experiment.encoder.out_feature_dim)
        else:
            self.bottleneck = nn.Identity()
        
    def forward(self, x):
        """Extract features from points."""
        
        x = self.vit(x)
        x = self.bottleneck(x[:, 1:,:])
        return x