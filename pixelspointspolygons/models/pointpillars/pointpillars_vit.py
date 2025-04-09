import logging
import timm

import open3d.ml.torch as ml3d

# from pointpillars_encoder import PointPillarsEncoder
from .pointpillars_ori import PointPillarsEncoder

from ...misc.logger import make_logger

class PointPillarsViT:
    
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

        self.vision_transformer = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        )
        # replace VisionTransformer patch embedding with LiDAR encoder
        self.vision_transformer.patch_embed = PointPillarsEncoder(cfg, local_rank=local_rank)
