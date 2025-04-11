import torch
import logging
import timm


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

    def __init__(self, cfg, local_rank=0):
        super().__init__()
        
        self.cfg = cfg        
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        self.vit = timm.create_model(
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
        self.vit.patch_embed = PointPillarsEncoder(cfg, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        
        
    def forward(self, x_lidar):
        """Extract features from points."""
        
        return self.vit(x_lidar)
