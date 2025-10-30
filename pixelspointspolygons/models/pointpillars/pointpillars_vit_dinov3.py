import torch
import logging
import timm
import os

import torch.nn as nn

# from .pointpillars_ori import PointPillarsEncoder
# from .pointpillars_o3d import PointPillarsEncoderOpen3D
from .pointpillars_ori import PointPillarsEncoder

from ...misc.logger import make_logger

class PointPillarsViTDINOv3(torch.nn.Module):
    
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

        # Safety check
        checkpoint_path = cfg.experiment.encoder.checkpoint_file
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
        
        # Step 1: Load model architecture
        self.vit = torch.hub.load(self.cfg.host.dino_v3_repo, self.cfg.experiment.encoder.type, pretrained=False, source='local')

        # Step 2: Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=self.cfg.host.device)

        # If checkpoint has extra keys (like 'model' or 'state_dict'), unwrap it
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Step 3: Load state dict into the model
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        self.logger.debug(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.debug(f"Missing keys: {missing}")
        self.logger.debug(f"Unexpected keys: {unexpected}")
        
        #### replace VisionTransformer patch embedding with LiDAR encoder        
        output_shape = [cfg.experiment.encoder.feature_map_width, cfg.experiment.encoder.feature_map_height]
        voxel_encoder={
            'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
            'feat_channels': [64,cfg.experiment.encoder.patch_feature_dim],
        }
        scatter={
            "in_channels" : cfg.experiment.encoder.patch_feature_dim, 
            "output_shape" : output_shape
        }
        # self.vit.patch_embed = PointPillarsEncoderOpen3D(cfg, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        self.vit.patch_embed = PointPillarsEncoder(cfg, flatten_embedding=False, voxel_encoder=voxel_encoder, scatter=scatter, local_rank=local_rank)
        
        if bottleneck:
            self.bottleneck = nn.AdaptiveAvgPool1d(cfg.experiment.model.decoder.in_feature_dim)
        else:
            self.bottleneck = nn.Identity()
        
    def forward(self, x):
        """Extract features from points."""
        
        patch_tokens = self.vit.forward_features(x)["x_norm_patchtokens"]
        patch_tokens = self.bottleneck(patch_tokens)
        return patch_tokens