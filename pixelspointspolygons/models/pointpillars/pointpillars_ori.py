import torch
import logging
from torch import nn
from torch.nn import functional as F

import numpy as np

from ...misc.logger import make_logger

from .voxelization.voxelization import Voxelization


class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels=64, output_shape=[496, 432]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    #@auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.in_channels,
                                 self.nx * self.ny,
                                 dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # # # Undo the column stacking to final 4-dim tensor
        # batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
        #                                  self.nx)

        return batch_canvas
    
class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self, in_channels, out_channels, last_layer=False, mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    #@auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(dim=1,
                          keepdim=True) / num_voxels.type_as(inputs).view(
                              -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int,
                           device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.0, 40, 1).
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 voxel_size=(0.16, 0.16, 4),
                 point_cloud_range=(0, -40.0, -3, 70.0, 40.0, 1)):

        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0

        # with cluster center (+3) + with voxel center (+2)
        in_channels += 5

        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters,
                         out_filters,
                         last_layer=last_layer,
                         mode='max'))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        f_center = features[:, :, :2].clone().detach()
        f_center[:, :, 0] = f_center[:, :, 0] - (
            coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
            self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (
            coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
            self.y_offset)

        features_ls.append(f_center)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(dim=1)

class PointPillarsEncoder(nn.Module):
    
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

    def __init__(self, cfg, voxel_encoder, scatter, flatten_embedding=True, local_rank=0):
        
        super(PointPillarsEncoder, self).__init__()

        self.cfg = cfg

        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)
        
        self.flatten_embedding = flatten_embedding
        
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
        
        # self.voxel_layer = PointPillarsVoxelization(
        #     point_cloud_range=point_cloud_range, **voxelize)
        self.voxel_layer = Voxelization(point_cloud_range=point_cloud_range, deterministic=self.cfg.run_type=="debug",
                                        **voxelize
                                        )
        
        self.voxel_encoder = PillarFeatureNet(
            # voxel_size=voxel_encoder['voxel_size'],
            **voxel_encoder,
            point_cloud_range=point_cloud_range
        )
        
        self.middle_encoder = PointPillarsScatter(**scatter)
        
    def compute_density(self, x_lidar):
        
        x_lidar = list(torch.unbind(x_lidar, dim=0))
        
        n_points = 0
        for tensor in x_lidar:
            n_points += tensor.shape[0]
        
        density = n_points / (len(x_lidar)*56**2)
        
        self.logger.warning(f"{density}")
        
    @torch.no_grad()
    def voxelize(self, points):
        """Copied from o3d---pointpillars.py"""
        
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
        
    def forward(self, x_lidar):
        """Extract features from points."""
        
        # self.compute_density(x_lidar)
        
        # x_lidar = list(torch.unbind(x_lidar, dim=0))
        # voxels, num_points, coors = self.voxel_layer(x_lidar)
        
        voxels, num_points, coors = self.voxelize(x_lidar)
        
        ## This is just for debugging, to allow comparison with open3d ml implementation        
        # coors_np = coors.detach().cpu().numpy()

        # # Get lexicographic sorted indices
        # indices = np.lexsort((coors_np[:, 3], coors_np[:, 2], coors_np[:, 1], coors_np[:, 0]))
        
        # # Reorder tensors
        # voxels = voxels[indices]
        # num_points = num_points[indices]
        # coors = coors[indices]
        
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = x_lidar.shape[0] # WARNING: do not use self.cfg.experiment.model.batch_size here, because it can be wrong for truncated batches at the end of the loader in drop_last=False, e.g. in validation and testing
        x = self.middle_encoder(voxel_features, coors, batch_size)
        
        x = x.transpose(1, 2)
        
        if not self.flatten_embedding:

            x = x.view(batch_size, 
                       self.middle_encoder.ny, self.middle_encoder.nx,
                       self.middle_encoder.in_channels)
        return x
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class PointPillarsNet(nn.Module):
    
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
        
        raise NotImplementedError("This still needs to be updated to the ori PointPillars instead of using o3d ml. Basically just need to copy"
                                  " the head and neck pointpillars class from open3d ml and past them in this file.")
        
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
        super(PointPillarsNet,self).__init__(
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