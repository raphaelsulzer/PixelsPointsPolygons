import torch

from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
from torch.nn.parallel import DistributedDataParallel as DDP

from ..pointpillars import PointPillarsViT
from ..vision_transformer import ViT
from ..fusion_models.fusion_vit import FusionViT

def generate_square_subsequent_mask(sz,device):
    mask = (
        torch.triu(torch.ones((sz, sz), device=device)) == 1
    ).transpose(0, 1)

    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

    return mask

def create_mask(tgt, pad_idx):
    """
    tgt shape: (N, L)
    """

    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device=tgt.device)
    # changing the type here from bool to float32 to get rid of the torch warning
    tgt_padding_mask = (tgt == pad_idx).to(dtype=tgt_mask.dtype)

    return tgt_mask, tgt_padding_mask


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
        """This is only used in inference mode"""
        
               
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

    def forward(self, x_images, x_lidar, y):
        
        if self.cfg.use_images and not self.cfg.use_lidar:
            features = self.encoder(x_images)
        elif not self.cfg.use_images and self.cfg.use_lidar:
            features = self.encoder(x_lidar)
        elif self.cfg.use_images and self.cfg.use_lidar:
            features = self.encoder(x_images, x_lidar)
        else:
            raise ValueError("At least one of use_images or use_lidar must be True")
                
        preds, feats = self.decoder(features, y)
        perm_mat1 = self.scorenet1(feats)
        perm_mat2 = self.scorenet2(feats)
        perm_mat = perm_mat1 + torch.transpose(perm_mat2, 1, 2)

        perm_mat = log_optimal_transport(
            perm_mat, self.bin_score, self.sinkhorn_iterations
        )[:, : perm_mat.shape[1], : perm_mat.shape[2]]
        perm_mat = F.softmax(perm_mat, dim=-1)

        return preds, perm_mat

    
    def predict(self, encoded_image, tgt):
        """This is only used in inference mode"""
        
        # encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoded_image, tgt)
        return preds, feats
    


class Pix2PolyModel(torch.nn.Module):
    
    def __new__(self, cfg, vocab_size, local_rank):
        
        self.cfg = cfg
                
        if self.cfg.use_images and self.cfg.use_lidar:
            if self.cfg.encoder.name == "fusion_vit":
                encoder = FusionViT(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.encoder.name} not implemented for {self.__name__}")
            
        elif self.cfg.use_images:
            
            if self.cfg.encoder.name == "vit":
                encoder = ViT(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.encoder.name} not implemented for {self.__name__}")
            
        elif self.cfg.use_lidar: 
            
            if self.cfg.encoder.name == "pointpillars_vit":
                encoder = PointPillarsViT(self.cfg,local_rank=local_rank)
            else:
                raise NotImplementedError(f"Encoder {self.cfg.encoder.name} not implemented for {self.__name__}")
            
        else:
            raise ValueError("Please specify either and image or lidar encoder with encoder=<name>. See help for a list of available encoders.")
        
        decoder = Decoder(
            vocab_size=vocab_size,
            encoder_len=self.cfg.encoder.num_patches,
            dim=self.cfg.encoder.out_feature_dim,
            num_heads=8,
            num_layers=6,
            max_len=self.cfg.model.tokenizer.max_len,
            pad_idx=self.cfg.model.tokenizer.pad_idx,
        )
        model = EncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            cfg=self.cfg
        )
        model.to(self.cfg.device)
        
        if self.cfg.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
        return model