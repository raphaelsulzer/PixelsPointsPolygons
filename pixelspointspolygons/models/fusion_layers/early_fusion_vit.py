import torch
import logging
import timm
import os

import torch.nn as nn

# from .pointpillars_ori import PointPillarsEncoder
from ..pointpillars.pointpillars_o3d import PointPillarsEncoder

from ...misc.logger import make_logger

class EarlyFusionViT(torch.nn.Module):
    
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
        self.local_rank = local_rank

        ###### LiDAR encoder #######
        
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
        self.lidar_embed = PointPillarsEncoder(cfg, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        
        if not os.path.isfile(cfg.experiment.encoder.vit.checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file {cfg.experiment.encoder.vit.checkpoint_file} not found.")
        logging.getLogger('timm').setLevel(logging.WARNING)
        ###### Image encoder #######
        self.vit = timm.create_model(
            model_name=cfg.experiment.encoder.vit.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.experiment.encoder.vit.pretrained,
            checkpoint_path=cfg.experiment.encoder.vit.checkpoint_file
        )
        
        self.image_embed = self.vit.patch_embed
        self.image_embed.flatten = False

        self.vit.patch_embed = nn.Identity()
        
        # if self.cfg.experiment.lidar_dropout is None:
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(self.cfg.experiment.encoder.patch_feature_dim*2, self.cfg.experiment.encoder.patch_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.experiment.encoder.patch_feature_dim),
            nn.ReLU(inplace=True)
        )
        #### OPTIONAL: Add a mask channel to the fusion layer
        # else:
        #     self.logger.info(f"Using LiDAR dropout with probability {self.cfg.experiment.lidar_dropout}")
        #     self.fusion_layer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.cfg.experiment.encoder.patch_feature_dim * 2 + 1,
        #         out_channels=self.cfg.experiment.encoder.patch_feature_dim,
        #         kernel_size=3,
        #         padding=1
        #     ),
        #     nn.BatchNorm2d(self.cfg.experiment.encoder.patch_feature_dim),
        #     nn.ReLU(inplace=True)
        #     )
        
        self.bottleneck = nn.AdaptiveAvgPool1d(cfg.experiment.encoder.out_feature_dim)
    
    def forward(self, x_image, x_lidar):
        """Extract features from points."""
        
        x_image = self.image_embed(x_image)
        x_lidar = self.lidar_embed(x_lidar,return_flattened=False)
        
        #### OPTIONAL: Add a mask channel
        # if self.cfg.experiment.lidar_dropout is not None:
        #     if torch.rand(1).item() < self.cfg.experiment.lidar_dropout:
        #         x_lidar = torch.zeros_like(x_lidar)
        #         lidar_mask = torch.zeros((x_lidar.size(0), 1, x_lidar.size(2), x_lidar.size(3)), device=x_lidar.device)
        #     else:
        #         lidar_mask = torch.ones((x_lidar.size(0), 1, x_lidar.size(2), x_lidar.size(3)), device=x_lidar.device)
        #     x = torch.cat((x_image, x_lidar, lidar_mask), dim=1)
        # else:
        #     x = torch.cat((x_image, x_lidar), dim=1)
        
        if self.cfg.experiment.lidar_dropout is not None:

            apply_dropout = torch.rand(1, device=x_lidar.device)
            apply_dropout = apply_dropout.item() < self.cfg.experiment.lidar_dropout
            if apply_dropout:
                self.logger.debug(f"LiDAR feature dropout applied")
                x_lidar = x_lidar * 0.0
                
        x = torch.cat((x_image, x_lidar), dim=1)
        
        x = self.fusion_layer(x).flatten(2).transpose(1, 2)
        x = self.vit(x)[:, 1:, :]
        x = self.bottleneck(x)
        
        return x
