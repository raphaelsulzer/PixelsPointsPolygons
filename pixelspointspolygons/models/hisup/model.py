import torch

import torch.nn.functional as F

from math import log
from torch import nn
from torch.utils.data.dataloader import default_collate

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

class Encoder:
    def __init__(self, cfg):
        self.target_h = cfg.model.encoder.input_height
        self.target_w = cfg.model.encoder.input_width

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
    def __init__(self, cfg):
        super().__init__()

        self.annotation_encoder = Encoder(cfg)

        self.pred_height = cfg.model.encoder.output_height
        self.pred_width = cfg.model.encoder.output_width
        self.origin_height = cfg.model.encoder.input_height
        self.origin_width = cfg.model.encoder.input_width

        dim_in = cfg.model.encoder.out_feature_channels
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

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