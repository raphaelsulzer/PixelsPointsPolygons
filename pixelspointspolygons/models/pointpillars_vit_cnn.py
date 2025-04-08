import torch.nn as nn

from multitask_head import MultitaskHead
from pointpillars_vit import PointPillarsViT

class LiDAREncoder(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.vision_transformer = PointPillarsViT(cfg)
                                
        self.proj = nn.Sequential(
            nn.Upsample(size=128, mode='bilinear', align_corners=False),
            nn.Conv2d(self.cfg.encoder.patch_embed_dim, self.cfg.model.decoder.in_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.cfg.model.decoder.in_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.head = MultitaskHead(self.cfg.model.decoder.in_feature_dim, 2, head_size=[[2]])


    def forward(self, points):
        
        x = self.vision_transformer(points)
        
        x = x[:, 1:,:] # drop CLS token
        
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        x = self.proj(x)
        
        return x
