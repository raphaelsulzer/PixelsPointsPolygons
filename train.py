# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


import torch
from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup
import hydra
from omegaconf import OmegaConf

from tokenizer import Tokenizer
from utils import seed_everything, load_checkpoint, compute_dynamic_cfg_vars
from datasets.build_datasets import get_train_loader, get_val_loader

from models.model import Encoder, Decoder, EncoderDecoder
from engine import train_eval



def get_model(cfg,tokenizer):
    
    encoder = Encoder(model_name=cfg.model.type, pretrained=True, out_dim=256)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        encoder_len=cfg.model.num_patches,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_len=cfg.model.tokenizer.max_len,
        pad_idx=cfg.model.tokenizer.pad_idx,
    )
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        n_vertices=cfg.model.tokenizer.n_vertices,
        sinkhorn_iterations=cfg.model.sinkhorn_iterations
    )
    model.to(cfg.device)
    
    return model



def run_training(cfg):
    
    seed_everything(42)

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=cfg.model.tokenizer.num_bins,
        width=cfg.model.input_width,
        height=cfg.model.input_height,
        max_len=cfg.model.tokenizer.max_len
    )
    compute_dynamic_cfg_vars(cfg,tokenizer)

    model = get_model(cfg,tokenizer)

    train_loader = get_train_loader(cfg,tokenizer)
    val_loader = get_val_loader(cfg,tokenizer)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params/10**6:.2f}M parameters")

    weight = torch.ones(cfg.model.tokenizer.pad_idx + 1, device=cfg.device)
    weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.model.tokenizer.pad_idx, label_smoothing=cfg.model.label_smoothing, weight=weight)
    perm_loss_fn = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.model.learning_rate, weight_decay=cfg.model.weight_decay, betas=(0.9, 0.95))

    num_training_steps = cfg.model.num_epochs * (len(train_loader.dataset) // cfg.model.batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    if cfg.checkpoint is not None:
        start_epoch = load_checkpoint(
            torch.load(cfg.checkpoint),
            model,
            optimizer,
            lr_scheduler
        )
        cfg.model.start_epoch = start_epoch + 1

    train_eval(
        model,
        train_loader,
        val_loader,
        tokenizer,
        vertex_loss_fn,
        perm_loss_fn,
        optimizer,
        lr_scheduler=lr_scheduler,
        step='batch',
        cfg=cfg
    )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    run_training(cfg)

if __name__ == "__main__":
    main()