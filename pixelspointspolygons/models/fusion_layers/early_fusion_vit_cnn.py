import torch
import logging
import timm

import torch.nn as nn

# from .pointpillars_ori import PointPillarsEncoder
from ..pointpillars.pointpillars_o3d import PointPillarsEncoder
from timm.models import VisionTransformer

from ...misc.logger import make_logger

class EarlyFusionViTCNN(torch.nn.Module):
    
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
        self.pp_vit = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained,
            checkpoint_path=cfg.encoder.checkpoint_file
        )
        
        #### replace VisionTransformer patch embedding with LiDAR encoder        
        output_shape = [cfg.encoder.patch_feature_width, cfg.encoder.patch_feature_height]
        voxel_encoder={
            'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
            'feat_channels': [64,cfg.encoder.patch_feature_dim],
        }
        scatter={
            "in_channels" : cfg.encoder.patch_feature_dim, 
            "output_shape" : output_shape
        }
        self.lidar_embed = PointPillarsEncoder(cfg, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        
        ###### Image encoder #######
        self.vit = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained,
            checkpoint_path=cfg.encoder.checkpoint_file
        )
        
        self.image_embed = self.vit.patch_embed
        self.image_embed.flatten = False

        self.vit.patch_embed = nn.Identity()
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(self.cfg.encoder.patch_feature_dim*2, self.cfg.encoder.patch_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.encoder.patch_feature_dim),
            nn.ReLU(inplace=True)
        )

        self.proj = nn.Sequential(
            nn.Upsample(size=self.cfg.encoder.out_feature_size, mode='bilinear', align_corners=False),
            nn.Conv2d(self.cfg.encoder.patch_feature_dim, self.cfg.model.decoder.in_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.model.decoder.in_feature_dim),
            nn.ReLU(inplace=True)
        )

        
        
    def forward(self, x_image, x_lidar):
        """Extract features from points."""
        
        x_image = self.image_embed(x_image)
        x_lidar = self.lidar_embed(x_lidar,return_flattened=False)
        x = torch.cat((x_image, x_lidar), dim=1)
        
        x = self.fusion_layer(x).flatten(2).transpose(1, 2)
        
        x = self.vit(x)
        
        x = x[:, 1:,:] # drop CLS token        
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.proj(x)
        
        return x
