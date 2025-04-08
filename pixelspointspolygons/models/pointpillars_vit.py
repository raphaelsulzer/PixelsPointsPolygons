import torch
import timm

import open3d.ml.torch as ml3d

from pointpillars_encoder import PointPillarsEncoder

class PointPillarsViT(ml3d.models.PointPillars):
    
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

    def __init__(self,cfg):
        
        self.cfg = cfg

        self.point_pillars_encoder = PointPillarsEncoder(cfg)
                
        self.vision_transformer = timm.create_model(
            model_name=cfg.model.lidar_encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        )
        # replace VisionTransformer patch embedding with LiDAR encoder
        self.vision_transformer.patch_embed = self.point_pillars_encoder
        
        
    def forward(self, x_lidar):
        """Extract features from points."""
        
        return self.vision_transformer(x_lidar)
