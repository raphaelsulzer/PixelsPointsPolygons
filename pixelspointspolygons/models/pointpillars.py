import open3d.ml.torch as ml3d

class PointPillarsNoHead(ml3d.models.PointPillars):
    
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
        
        
        # TODO: might be cleaner to simply replace the head with nn.Identity() instead of removing all the unused modules
        
        # see here for allowed params: https://github.com/isl-org/Open3D-ML/blob/fcf97c07bf7a113a47d0fcf63760b245c2a2784e/ml3d/configs/pointpillars_lyft.yml
        point_cloud_range = [0, 0, 0, 
                             cfg.encoder.in_width, cfg.encoder.in_height, cfg.encoder.in_voxel_size.z]
        voxel_size = list(cfg.encoder.in_voxel_size.values())
        
        
        output_shape = [cfg.encoder.out_feature_width, cfg.encoder.out_feature_height]
        
        # max_voxels = [(cfg.encoder.input_size // cfg.encoder.patch_size)**2] * 2
        
        voxelize={
            'max_num_points': cfg.encoder.max_num_points_per_voxel,
            'voxel_size': voxel_size,
            'max_voxels': [cfg.encoder.max_num_voxels.train, cfg.encoder.max_num_voxels.test], 
        }
        voxel_encoder={
            'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
            'feat_channels': [64],
            'voxel_size': voxel_size
        }
        scatter={
            "in_channels" : 64, 
            "output_shape" : output_shape
        }
        augment={
            "PointShuffle": True
        }
            
        super(PointPillarsNoHead,self).__init__(
                 device=cfg.device,
                 num_input_features=3,
                 point_cloud_range=point_cloud_range,
                 voxelize=voxelize,
                 voxel_encoder=voxel_encoder,
                 scatter=scatter,
                 augment=augment)
        
        self.cfg = cfg
        
        # remove unsused modules from PointPillars
        del self.bbox_head
        del self.loss_cls
        del self.loss_bbox
        del self.loss_dir


        
    def forward(self, x_lidar):
        """Extract features from points."""
        
        # x_lidar = list(torch.unbind(x_lidar, dim=0))
        # print(f"x_lidar shape {x_lidar.shape}")
        # print(f"x_lidar dtype {x_lidar.dtype}")
        # print(f"x_lidar device {x_lidar.device}")
        voxels, num_points, coors = self.voxelize(x_lidar)
        x = self.voxel_encoder(voxels, num_points, coors)
        batch_size = x_lidar.shape[0] # WARNING: do not use self.cfg.model.batch_size here, because it can be wrong for truncated batches at the end of the loader in drop_last=False, e.g. in validation and testing
        x = self.middle_encoder(x, coors, batch_size)
        
        x = self.backbone(x)
        x = self.neck(x)
        
        return x