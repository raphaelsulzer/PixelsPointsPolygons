import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torchvision.models.segmentation._utils import _SimpleSegmentationModel

from ..pointpillars import *

from .unet_resnet import UNetResNetBackbone


def get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        i += 1
    # If we get out of the loop but out_channels is None, then the prev child of the parent module will be checked, etc.
    return out_channels


class EncoderDecoder(torch.nn.Module):
    def __init__(self, cfg, encoder):
        """

        :param config:
        :param backbone: A _SimpleSegmentationModel network, its output features will be used to compute seg and framefield.
        """
        super().__init__()
        assert cfg.encoder.compute_seg or cfg.encoder.compute_crossfield, \
            "Model has to compute at least one of those:\n" \
            "\t- segmentation\n" \
            "\t- cross-field"
        if cfg.use_images and not cfg.use_lidar:
            assert isinstance(encoder, _SimpleSegmentationModel), \
                "backbone should be an instance of _SimpleSegmentationModel"
        elif cfg.use_lidar and not cfg.use_images:
            assert isinstance(encoder, PointPillarsEncoder), \
                "backbone should be an instance of PointPillarsEncoder"
        elif cfg.use_images and cfg.use_lidar:
            assert isinstance(encoder, torch.nn.Module), \
                "backbone should be an instance of torch.nn.Module"
        else:
            raise ValueError("At least one of use_image or use_lidar must be True")
        
        self.cfg = cfg
        self.backbone = encoder

        # backbone_out_features = get_out_channels(self.backbone)
        backbone_out_features = self.cfg.encoder.out_feature_channels

        # --- Add other modules if activated in config:
        seg_channels = 0
        if self.cfg.encoder.compute_seg:
            seg_channels = self.cfg.encoder.seg.compute_vertex\
                           + self.cfg.encoder.seg.compute_edge\
                           + self.cfg.encoder.seg.compute_interior
            self.seg_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, seg_channels, 1),
                torch.nn.Sigmoid(),)

        if self.cfg.encoder.compute_crossfield:
            crossfield_channels = 4
            self.crossfield_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features + seg_channels, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, crossfield_channels, 1),
                torch.nn.Tanh(),
            )

    
    def inference(self, image):
        outputs = {}

        if not image.shape[0] == 4:
            a=5

        # --- Extract features for every pixel of the image with a U-Net --- #
        backbone_features = self.backbone(image)["out"]

        if self.cfg.encoder.compute_seg:
            # --- Output a segmentation of the image --- #
            seg = self.seg_module(backbone_features)
            seg_to_cat = seg.clone().detach()
            backbone_features = torch.cat([backbone_features, seg_to_cat], dim=1)  # Add seg to image features
            outputs["seg"] = seg

        if self.cfg.encoder.compute_crossfield:
            # --- Output a cross-field of the image --- #
            crossfield = 2 * self.crossfield_module(backbone_features)  # Outputs c_0, c_2 values in [-2, 2]
            outputs["crossfield"] = crossfield

        return outputs

    # @profile
    def forward(self, xb):
        if self.cfg.use_images and not self.cfg.use_lidar:
            final_outputs = self.inference(xb["image"])
        elif self.cfg.use_lidar and not self.cfg.use_images:
            final_outputs = self.inference(xb["lidar"])
        elif self.cfg.use_images and self.cfg.use_lidar:
            raise NotImplementedError("MultiEncoderDecoder not implemented yet")
        else:
            raise ValueError("At least one of use_image or use_lidar must be True")
        return final_outputs, xb



class FFLModel(torch.nn.Module):
    
    def __new__(self, cfg, local_rank):
        
        self.cfg = cfg
                
        if self.cfg.use_images and self.cfg.use_lidar:
            encoder = MultiEncoderDecoder(self.cfg)
        elif self.cfg.use_images:
            encoder = UNetResNetBackbone(self.cfg)
            encoder = _SimpleSegmentationModel(encoder, classifier=torch.nn.Identity())
        elif self.cfg.use_lidar: 
            encoder = PointPillarsEncoder(self.cfg)
        else:
            raise ValueError("At least one of use_image or use_lidar must be True")
        
        model = EncoderDecoder(
            encoder=encoder,
            cfg=self.cfg
        )
                
        model.to(self.cfg.device)
        
        if self.cfg.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
        return model