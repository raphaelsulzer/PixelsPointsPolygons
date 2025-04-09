import torch
import logging

import open3d.ml.torch as ml3d

from ...misc.logger import make_logger

class PointPillarsEncoder(ml3d.models.PointPillars):
    
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

    def __init__(self, cfg, voxel_encoder, scatter, local_rank=0):
        
        self.cfg = cfg

        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)
        
        
        # see here for allowed params: https://github.com/isl-org/Open3D-ML/blob/fcf97c07bf7a113a47d0fcf63760b245c2a2784e/ml3d/configs/pointpillars_lyft.yml
        point_cloud_range = [0, 0, 0, 
                             cfg.encoder.in_width, cfg.encoder.in_height, cfg.encoder.in_voxel_size.z]
        voxel_size = list(cfg.encoder.in_voxel_size.values())
        
        voxelize={
            'max_num_points': cfg.encoder.max_num_points_per_voxel,
            'voxel_size': voxel_size,
            'max_voxels': [cfg.encoder.max_num_voxels.train, cfg.encoder.max_num_voxels.test], 
        }
        voxel_encoder["voxel_size"] = voxel_size 
        augment={
            "PointShuffle": True
        }
            
        super(PointPillarsEncoder,self).__init__(
                 device=cfg.device,
                 num_input_features=3,
                 point_cloud_range=point_cloud_range,
                 voxelize=voxelize,
                 voxel_encoder=voxel_encoder,
                 scatter=scatter,
                 augment=augment)
        

        # remove unsused modules from PointPillars
        # del self.backbone
        # del self.neck
        del self.bbox_head
        del self.loss_cls
        del self.loss_bbox
        del self.loss_dir
        
        
    def forward(self, x_lidar, return_flattened=True):
        """Extract features from points."""
        
        # x_lidar = list(torch.unbind(x_lidar, dim=0))
        voxels, num_points, coors = self.voxelize(x_lidar)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = x_lidar.shape[0] # WARNING: do not use self.cfg.model.batch_size here, because it can be wrong for truncated batches at the end of the loader in drop_last=False, e.g. in validation and testing
        x = self.middle_encoder(voxel_features, coors, batch_size)
        
        ## flatten patches, NCHW -> NLC. Needed to pass directly to next layer of VisionTransformer (self.vit)
        
        # TODO: there is an unnessary opertation when combining this with Vit:
        # self.middle_encoder already has the correct format for the VisionTransformer, but the last operation
        # is batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)
        # which I just undo below whith x = x.flatten(2).transpose(1, 2)
        
        if return_flattened:
            return x.flatten(2).transpose(1, 2)
        else:
            return x
