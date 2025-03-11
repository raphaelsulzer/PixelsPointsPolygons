# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
from torch import nn
from torch import optim
from torch import distributed

from transformers import get_linear_schedule_with_warmup
import hydra
from omegaconf import OmegaConf

from tokenizer import Tokenizer
from utils import seed_everything, load_checkpoint, compute_dynamic_cfg_vars
from datasets.build_datasets import get_train_loader, get_val_loader

from models.model import Encoder, Decoder, EncoderDecoder
from engine import train_eval

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
    dist_url = "env://"  # default

    # only works with torch.distributed.launch or torch.run.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    distributed.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # this will make all .cuda() calls work properly.
    torch.cuda.set_device(local_rank)

    # synchronizes all threads to reach this point before moving on.
    distributed.barrier()

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
    if cfg.multi_gpu:
        init_distributed()
    
    os.makedirs(cfg.output_dir,exist_ok=True)
    
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

    local_rank = int(os.environ["LOCAL_RANK"]) if cfg.multi_gpu else 0
    if cfg.checkpoint is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        start_epoch = load_checkpoint(
            torch.load(cfg.checkpoint, map_location=map_location),
            model,
            optimizer,
            lr_scheduler
        )
        cfg.model.start_epoch = start_epoch + 1

    # Convert BatchNorm in model to SyncBatchNorm.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Wrap model with distributed data parallel.
    if cfg.multi_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        
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