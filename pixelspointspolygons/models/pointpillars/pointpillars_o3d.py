import torch
import logging
from torch import nn

import open3d.ml.torch as ml3d

from ...misc.logger import make_logger

from ..multitask_head import MultitaskHead

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
                             cfg.experiment.encoder.in_width, cfg.experiment.encoder.in_height, cfg.experiment.encoder.in_voxel_size.z]
        voxel_size = list(cfg.experiment.encoder.in_voxel_size.values())
        
        voxelize={
            'max_num_points': cfg.experiment.encoder.max_num_points_per_voxel,
            'voxel_size': voxel_size,
            'max_voxels': [cfg.experiment.encoder.max_num_voxels.train, cfg.experiment.encoder.max_num_voxels.test], 
        }
        voxel_encoder["voxel_size"] = voxel_size 
        augment={
            "PointShuffle": True
        }
            
        super(PointPillarsEncoder,self).__init__(
                 device=cfg.host.device,
                 num_input_features=3,
                 point_cloud_range=point_cloud_range,
                 voxelize=voxelize,
                 voxel_encoder=voxel_encoder,
                 scatter=scatter,
                 augment=augment)
        

        # remove unsused modules from PointPillars
        del self.backbone
        del self.neck
        del self.bbox_head
        del self.loss_cls
        del self.loss_bbox
        del self.loss_dir
        
    def compute_density(self, x_lidar):
        
        x_lidar = list(torch.unbind(x_lidar, dim=0))
        
        n_points = 0
        for tensor in x_lidar:
            n_points += tensor.shape[0]
        
        density = n_points / (len(x_lidar)*56**2)
        
        self.logger.warning(f"{density}")
        
        
        
    def forward(self, x_lidar, return_flattened=True):
        """Extract features from points."""
        
        # self.compute_density(x_lidar)
        

        # x_lidar = list(torch.unbind(x_lidar, dim=0))
        voxels, num_points, coors = self.voxelize(x_lidar)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = x_lidar.shape[0] # WARNING: do not use self.cfg.experiment.model.batch_size here, because it can be wrong for truncated batches at the end of the loader in drop_last=False, e.g. in validation and testing
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



class PointPillars(ml3d.models.PointPillars):
    
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
        
        self.cfg = cfg

        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)
        
        
        # see here for allowed params: https://github.com/isl-org/Open3D-ML/blob/fcf97c07bf7a113a47d0fcf63760b245c2a2784e/ml3d/configs/pointpillars_lyft.yml
        point_cloud_range = [0, 0, 0, 
                             cfg.experiment.encoder.in_width, cfg.experiment.encoder.in_height, cfg.experiment.encoder.in_voxel_size.z]
        voxel_size = list(cfg.experiment.encoder.in_voxel_size.values())
        
        voxelize={
            'max_num_points': cfg.experiment.encoder.max_num_points_per_voxel,
            'voxel_size': voxel_size,
            'max_voxels': [cfg.experiment.encoder.max_num_voxels.train, cfg.experiment.encoder.max_num_voxels.test], 
        }
        ### output shape has to be defined like this!!
        output_shape = [cfg.experiment.encoder.in_width // cfg.experiment.encoder.in_voxel_size.x, cfg.experiment.encoder.in_height // cfg.experiment.encoder.in_voxel_size.y]
        voxel_encoder={
            'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
            'voxel_size': voxel_size,
            'feat_channels': [64],
        }
        scatter={
            "in_channels" : 64, 
            "output_shape" : output_shape
        }
        augment={
            "PointShuffle": True
        }
        backbone={
            "in_channels": 64,
            "out_channels": [64, 128, 256],
            "layer_nums": [3, 5, 5],
            "layer_strides": [2, 2, 2]
        }
        neck={
            "in_channels": [64, 128, 256],
            "out_channels": self.cfg.experiment.model.point_pillars.out_channels,
            "upsample_strides": self.cfg.experiment.model.point_pillars.upsample_strides,
            "use_conv_for_no_stride": False
        }
        
        ## careful, the stupid o3d pointpillars class overwrites self.cfg
        super(PointPillars,self).__init__(
                    device=cfg.host.device,
                    num_input_features=3,
                    point_cloud_range=point_cloud_range,
                    voxelize=voxelize,
                    voxel_encoder=voxel_encoder,
                    scatter=scatter,
                    backbone=backbone,
                    neck=neck,
                    augment=augment)
        

        if sum(cfg.experiment.model.point_pillars.out_channels) != cfg.experiment.model.decoder.in_feature_dim:
            self.reduce_dim = nn.Conv2d(in_channels=sum(cfg.experiment.model.point_pillars.out_channels), out_channels=cfg.experiment.model.decoder.in_feature_dim, kernel_size=1, stride=1, padding=0)
            self.reduce_dim = nn.Sequential(
                self.reduce_dim,
                nn.ReLU()
            )
        else:
            self.reduce_dim = nn.Identity()

        # TODO: the head should go in the decoder
        # if cfg.experiment.model.name != "ffl":
        #     self.head = MultitaskHead(cfg.experiment.model.decoder.in_feature_dim, 2, head_size=[[2]])
        
        # remove unsused modules from PointPillars
        del self.bbox_head
        del self.loss_cls
        del self.loss_bbox
        del self.loss_dir
        
        
    def forward(self, x_lidar):
        """Extract features from points."""
        
        # x_lidar = list(torch.unbind(x_lidar, dim=0))
        voxels, num_points, coors = self.voxelize(x_lidar)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = x_lidar.shape[0] # WARNING: do not use self.cfg.experiment.model.batch_size here, because it can be wrong for truncated batches at the end of the loader in drop_last=False, e.g. in validation and testing
        # batch_size = coors[-1, 0].item() + 1 ## that is how they do it inside o3d
        x = self.middle_encoder(voxel_features, coors, batch_size)
        
        x = self.backbone(x)
        
        x = self.neck(x)
        
        x = self.reduce_dim(x)
                                
        return x
