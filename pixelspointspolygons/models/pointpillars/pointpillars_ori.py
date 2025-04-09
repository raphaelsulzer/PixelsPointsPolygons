# import os
# import logging
# import torch
# import torch.nn as nn
# from pointpillars.model import PillarLayer, PillarEncoder, Backbone, Neck

# from ...misc.logger import make_logger

# # from ..hisup.bn_helper import BatchNorm2d_class
# from ..multitask_head import MultitaskHead

# class PointPillarsEncoder(nn.Module):
#     def __init__(self, cfg, local_rank=0):
#         super().__init__()
        
#         self.cfg = cfg
        
#         verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
#         self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

#         voxel_size=tuple(cfg.encoder.in_voxel_size.values())
#         point_cloud_range=[0,0,0,cfg.encoder.in_width,cfg.encoder.in_height,cfg.encoder.in_voxel_size.z]
#         max_voxels=tuple(cfg.encoder.max_num_voxels.values())
#         max_num_points = cfg.encoder.max_num_points_per_voxel

#         self.pillar_layer = PillarLayer(voxel_size=voxel_size,
#                                         point_cloud_range=point_cloud_range,
#                                         max_num_points=max_num_points,
#                                         max_voxels=max_voxels)

#         self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
#                                             point_cloud_range=point_cloud_range,
#                                             in_channel=8,
#                                             out_channel=64)

#         layer_strides = [1 if voxel_size[0] == 4 else 2, 2, 2]
#         self.pillar_backbone = Backbone(in_channel=64,
#                                  out_channels=[64, 128, 256],
#                                  layer_nums=[3, 5, 5],
#                                         layer_strides=layer_strides)

#         self.pillar_neck = Neck(in_channels=[64, 128, 256],
#                          upsample_strides=[1, 2, 4],
#                          out_channels=[128, 128, 128])

#         # # this is not really the head from PointPillars, I just give it that name because the parameters should be counted as backbone params
#         head_size = [[2]]
#         num_class = sum(sum(head_size, []))
#         self.head = MultitaskHead(input_channels=cfg.encoder.out_feature_dim,num_class=num_class,head_size=head_size)
    
    
    
#     def forward(self, batched_pts):
#         # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
#         #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
#         #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
#         pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

#         # print(f"Average number of pillars per sample: {pillars.shape[0]/len(batched_pts)}")
#         # print(f"Average number of points per pillar: {npoints_per_pillar.cpu().numpy().mean()}")

#         # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
#         # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
#         # npoints_per_pillar: (p1 + p2 + ... + pb, )
#         #                     -> pillar_features: (bs, out_channel, y_l, x_l)
#         pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

#         pillar_features = self.pillar_backbone(pillar_features)

#         pillar_features = self.pillar_neck(pillar_features)

#         return pillar_features
        