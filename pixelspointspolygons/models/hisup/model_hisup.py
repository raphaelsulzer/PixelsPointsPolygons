import torch
import cv2
import os

import torch.nn.functional as F
from copy import deepcopy
from huggingface_hub import hf_hub_download
from math import log
from torch import nn
from torch.utils.data.dataloader import default_collate
from skimage.measure import label, regionprops
from torch.nn.parallel import DistributedDataParallel as DDP

# from ..pointpillars import *
from ..pointpillars import PointPillarsEncoder, PointPillarsViTCNN, PointPillars
from ..vision_transformer import ViTCNN
from ..multitask_head import MultitaskHead
from ..fusion_layers import FusionViTCNN
from ..hrnet import HighResolutionNet as HRNet48v2
from .afm_module.afm_op import afm
from .polygon import get_pred_junctions, generate_polygon

def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(t/w)

    return loss.mean()

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y = self.avg_pool(x1 + x2)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1 ,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        out = self.out_conv(x2 * y.expand_as(x2))
        return out

class AnnotationEncoder:
    def __init__(self, cfg):
        self.target_h = cfg.encoder.in_height
        self.target_w = cfg.encoder.in_width

    def __call__(self, annotations):
        targets = []
        metas   = []
        for ann in annotations:
            t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets),metas

    def _process_per_image(self, ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']
        junc_tag = ann['juncs_tag']
        jmap = torch.zeros((height, width), device=device, dtype=torch.long)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)

        edges_positive = ann['edges_positive']
        if len(edges_positive) == 0:
            afmap = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
        else:
            lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
            shape_info = torch.IntTensor([[0, lines.size(0), height, width]])
            afmap, label = afm(lines, shape_info.cuda(), height, width)

        xint, yint = junctions[:,0].long(), junctions[:,1].long()
        off_x = junctions[:,0] - xint.float()-0.5
        off_y = junctions[:,1] - yint.float()-0.5
        
        if xint.min() < 0 or xint.max() >= width or yint.min() < 0 or yint.max() >= height:
            raise ValueError('Junctions out of bound')
        
        jmap[yint, xint] = junc_tag
        joff[0, yint, xint] = off_x
        joff[1, yint, xint] = off_y
        meta = {
            'junc': junctions,
            'junc_index': ann['juncs_index'],
            'bbox': ann['bbox'],
        }

        mask = ann['mask'].float()
        target = {
            'jloc': jmap[None],
            'joff': joff,
            'mask': mask[None],
            'afmap': afmap[0]
        }
        return target, meta

class EncoderDecoder(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        
        self.cfg = cfg

        self.annotation_encoder = AnnotationEncoder(cfg)

        self.encoder = encoder
        
        self.pred_height = cfg.encoder.out_feature_height
        self.pred_width = cfg.encoder.out_feature_width
        self.origin_height = cfg.encoder.in_height
        self.origin_width = cfg.encoder.in_width

        dim_in = cfg.model.decoder.in_feature_dim
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)
        self.joff_head = MultitaskHead(self.cfg.model.decoder.in_feature_dim, 2, head_size=[[2]])

        self.a2m_att = ECA(dim_in)
        self.a2j_att = ECA(dim_in)

        self.mask_predictor = self._make_predictor(dim_in, 2)
        self.jloc_predictor = self._make_predictor(dim_in, 3)
        self.afm_predictor = self._make_predictor(dim_in, 2)

        self.refuse_conv = self._make_conv(2, dim_in//2, dim_in)
        self.final_conv = self._make_conv(dim_in*2, dim_in, 2)
        

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
                    nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, dim_out, kernel_size=1),
                )
        return layer
    
    def init_loss_dict(self):
        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask': 0.0,
            'loss_afm' : 0.0,
            'loss_remask': 0.0
        }
        return loss_dict
    
    def forward(self, x_images, x_points, y):
        if self.training:
            return self.forward_train(x_images, x_points, y=y)
        else:
            return self.forward_val(x_images, x_points, y=y)
    
    
    def forward_common(self,x_images,x_lidar,y=None):
        
        if y is not None:
            targets, _ = self.annotation_encoder(y)
        else:
            targets = None
            
        if self.cfg.use_images and not self.cfg.use_lidar:
            features = self.encoder(x_images)
        elif not self.cfg.use_images and self.cfg.use_lidar:
            features = self.encoder(x_lidar)
        elif self.cfg.use_images and self.cfg.use_lidar:
            features = self.encoder(x_images, x_lidar)
        else:
            raise ValueError("At least one of use_images or use_lidar must be True")
        
        joff_pred = self.joff_head(features)

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        mask_att_feature = self.a2m_att(afm_feature, mask_feature)
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))
        
        return targets, joff_pred, jloc_pred, mask_pred, afm_pred, remask_pred

    
    def forward_val(self, x_images, x_lidar, y):
        
        targets, joff_pred, jloc_pred, mask_pred, afm_pred, remask_pred = self.forward_common(x_images, x_lidar, y)
        
        assert joff_pred.size(2) == self.pred_height
        assert joff_pred.size(3) == self.pred_width

        ##########################
        ######## val loss ########    
        ##########################
        loss_dict = self.init_loss_dict()
        if targets is not None:
            loss_dict['loss_jloc'] += F.cross_entropy(jloc_pred, targets['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += sigmoid_l1_loss(joff_pred[:, :], targets['joff'], -0.5, targets['jloc'])
            loss_dict['loss_mask'] += F.cross_entropy(mask_pred, targets['mask'].squeeze(dim=1).long())
            loss_dict['loss_afm'] += F.l1_loss(afm_pred, targets['afmap'])
            loss_dict['loss_remask'] += F.cross_entropy(remask_pred, targets['mask'].squeeze(dim=1).long())
        
        
        ##########################
        ##### polygonization #####    
        ##########################
        joff_pred = joff_pred[:, :].sigmoid() - 0.5
        jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]
        jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]
        remask_pred = remask_pred.softmax(1)[:, 1:]
        
        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(remask_pred.size(0)):
            mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])
            juncs_pred[:,0] *= scale_x
            juncs_pred[:,1] *= scale_y

            polys, scores = [], []
            props = regionprops(label(mask_pred_per_im > 0.5))
            for prop in props:
                poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, \
                                                                        juncs_pred, 0, False)
                if juncs_sa.shape[0] == 0:
                    continue

                polys.append(poly)
                scores.append(score)
            batch_scores.append(scores)
            batch_polygons.append(polys)

            batch_masks.append(mask_pred_per_im)
            batch_juncs.append(juncs_pred)

        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        
        return output, loss_dict


    def forward_train(self, x_images, x_lidar, y = None):

        targets, joff_pred, jloc_pred, mask_pred, afm_pred, remask_pred = self.forward_common(x_images,x_lidar,y)
        
        loss_dict = self.init_loss_dict()
        if targets is not None:
            loss_dict['loss_jloc'] += F.cross_entropy(jloc_pred, targets['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += sigmoid_l1_loss(joff_pred[:, :], targets['joff'], -0.5, targets['jloc'])
            loss_dict['loss_mask'] += F.cross_entropy(mask_pred, targets['mask'].squeeze(dim=1).long())
            loss_dict['loss_afm'] += F.l1_loss(afm_pred, targets['afmap'])
            loss_dict['loss_remask'] += F.cross_entropy(remask_pred, targets['mask'].squeeze(dim=1).long())

        return loss_dict



class HiSupModel(torch.nn.Module):
    
    def __new__(self, cfg, local_rank):
        
        self.cfg = cfg
                
        if self.cfg.use_images and self.cfg.use_lidar:
            
            if self.cfg.encoder.name == "fusion_hrnet":
                encoder = FusionHRNet(self.cfg,local_rank=local_rank)
            elif self.cfg.encoder.name == "fusion_vit_cnn":
                encoder = FusionViTCNN(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.encoder.name} not implemented for {self.__name__}")
            
            
        elif self.cfg.use_images:
            
            if self.cfg.encoder.name == "hrnet":
                encoder = HRNet48v2(self.cfg,local_rank=local_rank)
            elif self.cfg.encoder.name == "vit_cnn":
                encoder = ViTCNN(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.encoder.name} not implemented for {self.__name__}")
            
        elif self.cfg.use_lidar: 
            
            if self.cfg.encoder.name == "pointpillars":
                encoder = PointPillars(self.cfg,local_rank=local_rank)
            elif self.cfg.encoder.name == "pointpillars_vit_cnn":
                encoder = PointPillarsViTCNN(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.encoder.name} not implemented for {self.__name__}")
            
        else:
            raise ValueError("Please specify either and image or lidar encoder with encoder=<name>. See help for a list of available encoders.")
        
        model = EncoderDecoder(
            encoder=encoder,
            cfg=self.cfg
        )
        
        model.to(self.cfg.device)
        
        if self.cfg.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
        return model