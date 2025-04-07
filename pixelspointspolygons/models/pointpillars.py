import open3d.ml.torch as ml3d
import torch

class PointPillarsWithoutHead(ml3d.models.PointPillars):
    
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
                             cfg.model.lidar_encoder.in_width, cfg.model.lidar_encoder.in_height, cfg.model.lidar_encoder.in_voxel_size.z]
        voxel_size = list(cfg.model.lidar_encoder.in_voxel_size.values())
        
        
        output_shape = [cfg.model.lidar_encoder.out_feature_width, cfg.model.lidar_encoder.out_feature_height]
        
        # max_voxels = [(cfg.model.encoder.input_size // cfg.model.encoder.patch_size)**2] * 2
        
        voxelize={
            'max_num_points': cfg.model.lidar_encoder.max_num_points_per_voxel,
            'voxel_size': voxel_size,
            'max_voxels': [cfg.model.lidar_encoder.max_num_voxels.train, cfg.model.lidar_encoder.max_num_voxels.test], 
        }
        voxel_encoder={
            'in_channels': 3, # note that this is the number of input channels, o3d automatically adds the pillar features to this
            'feat_channels': [64,cfg.model.lidar_encoder.out_feature_dim],
            'voxel_size': voxel_size
        }
        scatter={
            "in_channels" : cfg.model.lidar_encoder.out_feature_dim, 
            "output_shape" : output_shape
        }
        augment={
            "PointShuffle": True
        }
            
        super(PointPillarsWithoutHead,self).__init__(
                 device=cfg.device,
                 num_input_features=3,
                 point_cloud_range=point_cloud_range,
                 voxelize=voxelize,
                 voxel_encoder=voxel_encoder,
                 scatter=scatter,
                 augment=augment)
        
        self.cfg = cfg

        # remove unsused modules from PointPillars
        del self.backbone
        del self.neck
        del self.bbox_head
        del self.loss_cls
        del self.loss_bbox
        del self.loss_dir
        
        
    def forward(self, x_lidar):
        """Extract features from points."""
        
        #x_lidar = list(torch.unbind(x_lidar, dim=0))
        # print(f"x_lidar shape {x_lidar.shape}")
        # print(f"x_lidar dtype {x_lidar.dtype}")
        # print(f"x_lidar device {x_lidar.device}")
        voxels, num_points, coors = self.voxelize(x_lidar)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = x_lidar.shape[0] # WARNING: do not use self.cfg.model.batch_size here, because it can be wrong for truncated batches at the end of the loader in drop_last=False, e.g. in validation and testing
        x = self.middle_encoder(voxel_features, coors, batch_size)

        # TODO: there is an unnessary opertation when combining this with Vit:
        # self.middle_encoder already has the correct format for the VisionTransformer, but the last operation
        # is batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)
        # which I just undo below whith x = x.flatten(2).transpose(1, 2)

        # if self.cfg.model.name == "pix2poly":
        #     # flatten patches, NCHW -> NLC. Needed to pass directly to next layer of VisionTransformer (self.vit)
        #     x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        # elif self.cfg.model.name == "hisup":
        #     pass
        # elif self.cfg.model.name == "ffl":
        #     x = {"out" : x}
        # else:
        #     raise NotImplementedError(f"Model {self.cfg.model.name} not implemented")
        
        if self.cfg.model.lidar_encoder.name == "pointpillars_vit":
            # flatten patches, NCHW -> NLC. Needed to pass directly to next layer of VisionTransformer (self.vit)
            x = x.flatten(2).transpose(1, 2)
        elif self.cfg.model.lidar_encoder.name == "pointpillars":
            pass
        else:
            raise NotImplementedError(f"Model {self.cfg.model.lidar_encoder.name} not implemented")        
        
        return x
