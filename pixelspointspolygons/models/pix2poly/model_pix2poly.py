import timm
import torch

from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

from ..pointpillars import PointPillarsEncoder

from .utils import create_mask


# Borrowed from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py#L143
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# Borrowed from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py#L152
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class ScoreNet(nn.Module):
    def __init__(self, n_vertices, in_channels=512):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feats):
        feats = feats[:, 1:]
        feats = feats.unsqueeze(2)
        feats = feats.view(feats.size(0), feats.size(1)//2, 2, feats.size(3))
        feats = torch.mean(feats, dim=2)

        x = torch.transpose(feats, 1, 2)
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, 1, self.n_vertices)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)

        return x[:, 0]


class LiDAREncoder(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_pillars = PointPillarsEncoder(cfg)        
        self.vision_transformer = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        )
        # replace VisionTransformer patch embedding with LiDAR encoder
        self.vision_transformer.patch_embed = self.point_pillars
                        
        self.bottleneck = nn.AdaptiveAvgPool1d(cfg.encoder.out_dim)


    def forward(self, x_images=None, x_lidar=None):
        
        # x = self.point_pillars(x_lidar)
        x = self.vision_transformer(x_lidar)
        x = self.bottleneck(x[:, 1:,:])
        
        return x
    

class ImageEncoder(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        )
        self.bottleneck = nn.AdaptiveAvgPool1d(cfg.encoder.out_dim)
    
    def forward(self, x_images=None, x_lidar=None):
        
        features = self.model(x_images)
        return self.bottleneck(features[:, 1:,:])


class FeatureFusionLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_pillars = PointPillarsEncoder(cfg)        
        self.vit_patch_embed = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        ).patch_embed                
                
        self.fusion = nn.Linear(cfg.encoder.patch_feature_dim*2, cfg.encoder.patch_feature_dim)

        
    def forward(self, x_images, x_lidar):
        
        x_lidar = self.point_pillars(x_lidar)
        x_images = self.vit_patch_embed(x_images)
        
        x = torch.cat([x_images, x_lidar], dim=-1)
        x = self.fusion(x)
        
        return x

class PatchFusionLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_pillars = PointPillarsEncoder(cfg)        
        self.vit_patch_embed = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        ).patch_embed

        
    def forward(self, x_images, x_lidar):
        
        x_lidar = self.point_pillars(x_lidar)
        x_images = self.vit_patch_embed(x_images)
        
        x = torch.cat([x_images, x_lidar], dim=1)
        
        return x


class MultiEncoder(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.multi_vision_transformer = timm.create_model(
            model_name=cfg.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.encoder.pretrained
        )
        # identity patch embedding, which is already done in fusion layer
        self.multi_vision_transformer.patch_embed = nn.Identity()

        num_patches = (cfg.encoder.input_size // cfg.encoder.patch_size)**2

        if cfg.model.fusion == "patch_concat":
            self.fusion_layer1 = PatchFusionLayer(cfg)
            self.fusion_layer2 = nn.Linear(num_patches*2,num_patches)
            
            modality_embed = nn.Embedding(2, cfg.encoder.patch_feature_dim)

            modality_ids = torch.cat([
                torch.zeros((1, 1), dtype=torch.long),      # CLS
                torch.zeros((1, num_patches), dtype=torch.long),      # image patches
                torch.ones((1, num_patches), dtype=torch.long)        # lidar patches
            ], dim=1)  # (1, 2L+1)


            # fix the pos_embeding to also account for lidar patches and then add the modality embedding
            self.multi_vision_transformer.pos_embed = \
                nn.Parameter(torch.cat([self.multi_vision_transformer.pos_embed, self.multi_vision_transformer.pos_embed[:,1:,:] ], dim=1)+ \
                    + modality_embed(modality_ids))
            
            self.forward = self.forward_patch_concat

            
        elif cfg.model.fusion == "feature_concat":
            self.fusion_layer1 = FeatureFusionLayer(cfg)
            self.forward = self.forward_feature_concat
        else:
            raise ValueError(f"Invalid fusion layer type {cfg.model.fusion} specified. Choose from 'patch_concat' or 'feature_concat'")
            
        self.bottleneck = nn.AdaptiveAvgPool1d(cfg.encoder.out_dim)


    def forward_patch_concat(self, x_images, x_lidar):
        
        x = self.fusion_layer1(x_images, x_lidar)        
        x = self.multi_vision_transformer(x)
        x = x.permute(0, 2, 1)
        x = self.fusion_layer2(x[:, :,1:]).permute(0, 2, 1)
        x = self.bottleneck(x)
        
        return x

    def forward_feature_concat(self, x_images, x_lidar):
        
        x = self.fusion_layer1(x_images, x_lidar)        
        x = self.multi_vision_transformer(x)
        x = self.bottleneck(x[:, 1:,:])
        
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_len: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_len: int,
        pad_idx: int,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.max_len - 1, dim) * 0.02
        )
        self.decoder_pos_drop = nn.Dropout(p=0.05)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_len, dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=0.05)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name:
                # print("Skipping initialization of pos embed layers...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

    def forward(self, encoder_out, tgt):
        """
        encoder_out shape: (N, L, D)
        tgt shape: (N, L)
        """

        tgt_mask, tgt_padding_mask = create_mask(tgt, self.pad_idx)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(
            memory=encoder_out,
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        preds = preds.transpose(0, 1)
        return self.output(preds), preds

    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = (
            torch.ones((tgt.size(0), self.max_len - length - 1), device=tgt.device)
            .fill_(self.pad_idx)
            .long()
        )
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt, self.pad_idx)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(
            memory=encoder_out,
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask = tgt_padding_mask
        )

        preds = preds.transpose(0, 1)
        return self.output(preds)[:, length-1, :], preds

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder
        self.n_vertices = cfg.model.tokenizer.n_vertices
        self.sinkhorn_iterations = cfg.model.sinkhorn_iterations
        self.scorenet1 = ScoreNet(self.n_vertices)
        self.scorenet2 = ScoreNet(self.n_vertices)
        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_image, x_lidar, y):
                
        encoder_out = self.encoder(x_image, x_lidar)
        preds, feats = self.decoder(encoder_out, y)
        perm_mat1 = self.scorenet1(feats)
        perm_mat2 = self.scorenet2(feats)
        perm_mat = perm_mat1 + torch.transpose(perm_mat2, 1, 2)

        perm_mat = log_optimal_transport(
            perm_mat, self.bin_score, self.sinkhorn_iterations
        )[:, : perm_mat.shape[1], : perm_mat.shape[2]]
        perm_mat = F.softmax(perm_mat, dim=-1)

        return preds, perm_mat

    
    def predict(self, encoded_image, tgt):
        # encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoded_image, tgt)
        return preds, feats