# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import logging
import hydra
from omegaconf import OmegaConf

import torch
from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup

from lidar_poly_dataset.misc import make_logger

from tokenizer import Tokenizer
from utils import seed_everything, load_checkpoint, compute_dynamic_cfg_vars
from ddp_utils import init_distributed
from datasets.build_datasets import get_train_loader, get_val_loader
from engine import train_eval
from models import get_model


class Trainer:
    def __init__(self,cfg,verbosity=logging.INFO):
        
        self.cfg = cfg
        self.logger = make_logger("Training",level=verbosity)
        self.logger.info("Create output directory {cfg.output_dir}...")
        os.makedirs(cfg.output_dir, exist_ok=True)
       

    def train(self):
        if self.cfg.multi_gpu:
            init_distributed()
                
        seed_everything(42)

        tokenizer = Tokenizer(
            num_classes=1,
            num_bins=self.cfg.model.tokenizer.num_bins,
            width=self.cfg.model.encoder.input_width,
            height=self.cfg.model.encoder.input_height,
            max_len=self.cfg.model.tokenizer.max_len
        )
        compute_dynamic_cfg_vars(self.cfg,tokenizer)

        model = get_model(self.cfg,tokenizer=tokenizer)

        train_loader = get_train_loader(self.cfg,tokenizer)
        val_loader = get_val_loader(self.cfg,tokenizer)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {n_params/10**6:.2f}M parameters")

        weight = torch.ones(self.cfg.model.tokenizer.pad_idx + 1, device=self.cfg.device)
        weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
        vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=self.cfg.model.tokenizer.pad_idx, label_smoothing=self.cfg.model.label_smoothing, weight=weight)
        perm_loss_fn = nn.BCELoss()

        optimizer = optim.AdamW(model.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))

        num_training_steps = self.cfg.model.num_epochs * (len(train_loader.dataset) // self.cfg.model.batch_size)
        num_warmup_steps = int(0.05 * num_training_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )

        local_rank = int(os.environ["LOCAL_RANK"]) if self.cfg.multi_gpu else 0
        if self.cfg.checkpoint is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            start_epoch = load_checkpoint(
                torch.load(self.cfg.checkpoint, map_location=map_location),
                model,
                optimizer,
                lr_scheduler
            )
            self.cfg.model.start_epoch = start_epoch + 1

        # Convert BatchNorm in model to SyncBatchNorm.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Wrap model with distributed data parallel.
        if self.cfg.multi_gpu:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        
        ## now store config to file, after all the dynamic variables have been computed
        config_save_path = os.path.join(self.cfg.output_dir, 'config.yaml')
        OmegaConf.save(config=self.cfg, f=config_save_path)
        self.logger.info(f"Configuration saved to {config_save_path}")
        
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
            cfg=self.cfg
        )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    tt = Trainer(cfg)
    tt.train()

if __name__ == "__main__":
    main()