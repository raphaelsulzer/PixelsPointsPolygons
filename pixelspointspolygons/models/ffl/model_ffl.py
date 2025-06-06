import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from transformers import ConvNextV2Model

from ..pointpillars import *
from ..vision_transformer import *
from ..hrnet import HighResolutionNet as HRNet48v2
from ..unetresnet.unet_resnet import UNetResNetBackbone
from ..fusion_layers import EarlyFusionViTCNN

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
        assert cfg.experiment.model.compute_seg or cfg.experiment.model.compute_crossfield, \
            "Model has to compute at least one of those:\n" \
            "\t- segmentation\n" \
            "\t- cross-field"
        
        self.cfg = cfg
        self.encoder = encoder

        # backbone_out_features = get_out_channels(self.backbone)
        backbone_out_features = self.cfg.experiment.encoder.out_feature_dim

        # --- Add other modules if activated in config:
        seg_channels = 0
        if self.cfg.experiment.model.compute_seg:
            seg_channels = self.cfg.experiment.model.seg.compute_vertex\
                           + self.cfg.experiment.model.seg.compute_edge\
                           + self.cfg.experiment.model.seg.compute_interior
            self.seg_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(backbone_out_features, seg_channels, 1),
                torch.nn.Sigmoid(),)

        if self.cfg.experiment.model.compute_crossfield:
            crossfield_channels = 4
            self.crossfield_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features + seg_channels, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(backbone_out_features, crossfield_channels, 1),
                torch.nn.Tanh(),
            )

    
    def inference(self, x_images, x_lidar):
        outputs = {}

        # --- Extract features for every pixel of the image with a U-Net --- #
        if self.cfg.experiment.encoder.use_images and not self.cfg.experiment.encoder.use_lidar:
            features = self.encoder(x_images)
        elif not self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
            features = self.encoder(x_lidar)
        elif self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
            features = self.encoder(x_images, x_lidar)
        else:
            raise ValueError("At least one of use_images or use_lidar must be True")

        if self.cfg.experiment.model.compute_seg:
            # --- Output a segmentation of the image --- #
            seg = self.seg_module(features)
            seg_to_cat = seg.clone().detach()
            features = torch.cat([features, seg_to_cat], dim=1)  # Add seg to image features
            outputs["seg"] = seg

        if self.cfg.experiment.model.compute_crossfield:
            # --- Output a cross-field of the image --- #
            crossfield = 2 * self.crossfield_module(features)  # Outputs c_0, c_2 values in [-2, 2]
            outputs["crossfield"] = crossfield

        return outputs

    # @profile
    def forward(self, x_batch):
        # TODO: the passing through of xb is a bit useless now since I removed the augmentations from the forward pass.
        # should be removed
        
        final_outputs = self.inference(x_batch.get("image",None), x_batch.get("lidar",None))
        return final_outputs



class FFLModel(torch.nn.Module):
    
    def __new__(self, cfg, local_rank):
        
        self.cfg = cfg
                
        if self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
            
            if self.cfg.experiment.encoder.name == "fusion_vit_cnn":
                encoder = FusionViTCNN(self.cfg,local_rank=local_rank)
            elif self.cfg.experiment.encoder.name == "early_fusion_vit_cnn":
                encoder = EarlyFusionViTCNN(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.experiment.encoder.name} not implemented for {self.__name__}")
            
            
        elif self.cfg.experiment.encoder.use_images:
            
            if self.cfg.experiment.encoder.name == "hrnet":
                encoder = HRNet48v2(self.cfg,local_rank=local_rank)
            elif self.cfg.experiment.encoder.name == "unetresnet101":
                encoder = UNetResNetBackbone(self.cfg)
                encoder = _SimpleSegmentationModel(encoder, classifier=torch.nn.Identity())
            elif self.cfg.experiment.encoder.name == "convnext":
                raise NotImplementedError(f"Encoder {self.cfg.experiment.encoder.name} not implemented for {self.__name__}")
            elif self.cfg.experiment.encoder.name == "convnext_v2":
                # all this needs is to be made into a class and put an upsampling function in the forwards pass
                encoder = ConvNextV2Model.from_pretrained(self.cfg.experiment.encoder.convnext.checkpoint_file)
                
            elif self.cfg.experiment.encoder.name == "vit_cnn":
                encoder = ViTCNN(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.experiment.encoder.name} not implemented for {self.__name__}")
            
        elif self.cfg.experiment.encoder.use_lidar: 
            
            if self.cfg.experiment.encoder.name == "pointpillars":
                encoder = PointPillars(self.cfg,local_rank=local_rank)
            elif self.cfg.experiment.encoder.name == "pointpillars_vit_cnn":
                encoder = PointPillarsViTCNN(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.experiment.encoder.name} not implemented for {self.__class__.__name__}")
            
        else:
            raise ValueError("At least one of use_image or use_lidar must be True")
        
        model = EncoderDecoder(
            encoder=encoder,
            cfg=self.cfg
        )
                
        model.to(self.cfg.host.device)
        
        if self.cfg.host.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
        return model