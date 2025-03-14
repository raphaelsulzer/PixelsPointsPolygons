import timm
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

import os
import sys
sys.path.insert(1, os.getcwd())

from ..misc import create_mask


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
    pass



class Encoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            model_name=cfg.model.encoder.type,
            num_classes=0,
            global_pool='',
            pretrained=cfg.model.encoder.pretrained
        )
        self.bottleneck = nn.AdaptiveAvgPool1d(cfg.model.encoder.out_dim)

    def forward(self, x_images, x_lidar):
        if self.cfg.use_images and self.cfg.use_lidar:
            return self.forward_both(x_images, x_lidar)
        elif self.cfg.use_images and not self.cfg.use_lidar:
            return self.forward_images(x_images)
        elif not self.cfg.use_images and self.cfg.use_lidar:
            return self.forward_lidar(x_lidar)
        else:
            raise ValueError("At least one of images or LiDAR must be used")
    
    def forward_images(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:,:])
    
    def forward_lidar(self, x):
        raise NotImplementedError("LiDAR encoder not implemented yet")
    
    def forward_both(self, x_images, x_lidar):
        
        return self.forward_images(x_images)
        a=5


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
        n_vertices: int,
        sinkhorn_iterations: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_vertices = n_vertices
        self.sinkhorn_iterations = sinkhorn_iterations
        self.scorenet1 = ScoreNet(self.n_vertices)
        self.scorenet2 = ScoreNet(self.n_vertices)
        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image, lidar, tgt):
        encoder_out = self.encoder(image, lidar)
        preds, feats = self.decoder(encoder_out, tgt)
        perm_mat1 = self.scorenet1(feats)
        perm_mat2 = self.scorenet2(feats)
        perm_mat = perm_mat1 + torch.transpose(perm_mat2, 1, 2)

        perm_mat = log_optimal_transport(
            perm_mat, self.bin_score, self.sinkhorn_iterations
        )[:, : perm_mat.shape[1], : perm_mat.shape[2]]
        perm_mat = F.softmax(perm_mat, dim=-1)

        return preds, perm_mat

    # def predict(self, image, tgt):
    #     encoder_out = self.encoder(image)
    #     preds, feats = self.decoder.predict(encoder_out, tgt)
    #     return preds, feats
    
    def predict(self, encoded_image, tgt):
        # encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoded_image, tgt)
        return preds, feats
    

if __name__ == "__main__":
    from pixelspointspolygons.models.tokenizer import Tokenizer
    from torch.nn.utils.rnn import pad_sequence
    import numpy as np
    import torch
    from torch import nn
    from dataclasses import dataclass

    @dataclass
    class DebugConfig:
        input_height: int = 224
        input_width: int = 224
        img_size: int = 224
        num_bins: int = 224
        max_len: int = 386
        n_vertices: int = 192
        sinkhorn_iterations: int = 100
        model_name: str = "vit_small_patch8_224_dino"
        num_patches: int = 784
        device: str = "cuda"

    def run_debug(config: DebugConfig):
        # Create sample input
        image = torch.randn(1, 3, config.input_height, config.input_width).to(
            config.device
        )

        # Generate random ground truth coordinates
        gt_coords = np.random.randint(
            size=(config.n_vertices, 2), low=0, high=config.img_size
        ).astype(np.float32)

        # Initialize tokenizer and process coordinates
        tokenizer = Tokenizer(
            num_classes=1,
            num_bins=config.num_bins,
            width=config.img_size,
            height=config.img_size,
            max_len=config.max_len,
        )
        gt_seqs, rand_idxs = tokenizer(gt_coords)

        # Prepare sequences for training
        gt_seqs = [torch.LongTensor(gt_seqs)]
        gt_seqs = pad_sequence(
            gt_seqs, batch_first=True, padding_value=tokenizer.PAD_code
        )
        pad = (
            torch.ones(gt_seqs.size(0), config.max_len - gt_seqs.size(1))
            .fill_(tokenizer.PAD_code)
            .long()
        )
        gt_seqs = torch.cat([gt_seqs, pad], dim=1).to(config.device)

        gt_seqs_input = gt_seqs[:, :-1]
        gt_seqs_expected = gt_seqs[:, 1:]

        # Initialize model components
        encoder = Encoder(model_name=config.model_name, pretrained=False, out_dim=256)
        decoder = Decoder(
            vocab_size=tokenizer.vocab_size,
            encoder_len=config.num_patches,
            dim=256,
            num_heads=8,
            num_layers=6,
            max_len=config.max_len,
            pad_idx=tokenizer.PAD_code,
        )
        model = EncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            n_vertices=config.n_vertices,
            sinkhorn_iterations=config.sinkhorn_iterations,
        ).to(config.device)

        vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_code)

        # Forward pass
        preds_f, perm_mat = model(image, gt_seqs_input)
        loss = vertex_loss_fn(
            preds_f.reshape(-1, preds_f.shape[-1]), gt_seqs_expected.reshape(-1)
        )

        # Test sequence generation
        batch_preds = (
            torch.ones(image.size(0), 1)
            .fill_(tokenizer.BOS_code)
            .long()
            .to(config.device)
        )

        def sample(preds):
            return torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

        confs = []
        with torch.no_grad():
            for i in range(1 + config.n_vertices * 2):
                try:
                    print(f"Generation step: {i}")
                    preds_p, feats_p = model.predict(image, batch_preds)
                    if i % 2 == 0:
                        confs_ = (
                            torch.softmax(preds_p, dim=-1)
                            .sort(axis=-1, descending=True)[0][:, 0]
                            .cpu()
                        )
                        confs.append(confs_)
                    preds_p = sample(preds_p)
                    batch_preds = torch.cat([batch_preds, preds_p], dim=1)
                except Exception as e:
                    print(f"Error at iteration {i}: {e}")
                    break

            # Process results
            EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
            invalid_idxs = ((EOS_idxs - 1) % 2 != 0).nonzero().view(-1)
            EOS_idxs[invalid_idxs] = 0

            all_coords = []
            all_confs = []
            for i, EOS_idx in enumerate(EOS_idxs.tolist()):
                if EOS_idx == 0:
                    all_coords.append(None)
                    all_confs.append(None)
                    continue
                coords = tokenizer.decode(batch_preds[i, : EOS_idx + 1])
                conf_vals = [round(confs[j][i].item(), 3) for j in range(len(coords))]
                all_coords.append(coords)
                all_confs.append(conf_vals)

        # Print debug info
        print("\nResults:")
        print(f"Predictions shape: {preds_f.shape}")
        print(f"Predictions require grad: {preds_f.requires_grad}")
        print(f"Predictions range: [{preds_f.min():.2f}, {preds_f.max():.2f}]")
        print(f"Permutation matrix shape: {perm_mat.shape}")
        print(f"Loss value: {loss.item():.4f}")

    debug_config = DebugConfig()
    run_debug(debug_config)