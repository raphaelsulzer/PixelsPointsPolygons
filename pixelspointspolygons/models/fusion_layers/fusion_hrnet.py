import torch
import logging
import timm

import torch.nn as nn

# from .pointpillars_ori import PointPillarsEncoder
from ..pointpillars import PointPillarsViT
from ..hrnet import HighResolutionNet

from ...misc.logger import make_logger

class FusionHRNet(torch.nn.Module):
    
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

        ###### LiDAR encoder #######
        self.pp_vit = PointPillarsViT(cfg,local_rank=local_rank)
        
        ###### Image encoder #######
        self.hrnet = HighResolutionNet(self.cfg,local_rank=local_rank)
        
        self.proj = nn.Sequential(
            nn.Upsample(size=self.cfg.encoder.out_feature_size, mode='bilinear', align_corners=False),
            nn.Conv2d(self.cfg.encoder.patch_feature_dim, self.cfg.encoder.patch_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.encoder.patch_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(self.cfg.encoder.patch_feature_dim+self.cfg.encoder.out_feature_dim, self.cfg.model.decoder.in_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.model.decoder.in_feature_dim),
            nn.ReLU(inplace=True)
        )

        
        
    def forward(self, x_image, x_lidar):
        """Extract features from points."""
        
        x_image = self.hrnet(x_image)
        x_lidar = self.pp_vit(x_lidar)
        
        B, N, C = x_lidar.shape
        H = W = int(N ** 0.5)
        
        x_lidar = x_lidar.permute(0, 2, 1).view(B, C, H, W)
        x_lidar = self.proj(x_lidar)
        
        x = torch.cat((x_image, x_lidar), dim=1)
        
        x = self.fusion_layer(x)
        
        
        return x
