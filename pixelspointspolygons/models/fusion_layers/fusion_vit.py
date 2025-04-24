import torch
import logging
import timm

import torch.nn as nn

# from .pointpillars_ori import PointPillarsEncoder
from ..pointpillars import PointPillarsViT

from ...misc.logger import make_logger

class FusionViT(torch.nn.Module):
    
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

    def __init__(self, cfg, local_rank=0):
        super().__init__()
        
        self.cfg = cfg        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        # ###### LiDAR encoder #######
        # self.pp_vit = timm.create_model(
        #     model_name=cfg.experiment.encoder.type,
        #     num_classes=0,
        #     global_pool='',
        #     pretrained=cfg.experiment.encoder.pretrained,
        #     checkpoint_path=cfg.experiment.encoder.checkpoint_file
        # )
        
        # #### replace VisionTransformer patch embedding with LiDAR encoder        
        # output_shape = [cfg.experiment.encoder.patch_feature_width, cfg.experiment.encoder.patch_feature_height]
        # voxel_encoder={
        #     'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
        #     'feat_channels': [64,cfg.experiment.encoder.patch_feature_dim],
        # }
        # scatter={
        #     "in_channels" : cfg.experiment.encoder.patch_feature_dim, 
        #     "output_shape" : output_shape
        # }
        # self.pp_vit.patch_embed = PointPillarsEncoder(cfg, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        self.pp_vit = PointPillarsViT(cfg)
        
        ###### Image encoder #######
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.vit.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.experiment.encoder.vit.pretrained,
            checkpoint_path=cfg.experiment.encoder.vit.checkpoint_file
        )
        
        
        # self.bottleneck = nn.AdaptiveAvgPool1d(cfg.experiment.encoder.out_feature_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.cfg.experiment.encoder.patch_feature_dim*2, self.cfg.experiment.model.decoder.in_feature_dim),  # Linear layer to reduce dimensionality
            nn.ReLU(),            # Non-linearity
            nn.LayerNorm(self.cfg.experiment.model.decoder.in_feature_dim)   # BatchNorm for stabilization (applied to tokens dimension)
        )

        
        
    def forward(self, x_image, x_lidar):
        """Extract features from points."""
        
        x_image = self.vit(x_image)[:, 1:,:]
        x_lidar = self.pp_vit(x_lidar)[:, 1:,:]
        
        x = torch.cat((x_image, x_lidar), dim=-1)
        
        x = self.fusion_layer(x)
        
        
        return x
