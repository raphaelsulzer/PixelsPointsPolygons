# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import sys
import os
import torch
import wandb
import json
import shutil
import logging
import torch

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import OmegaConf
from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup

from ..models.tokenizer import Tokenizer
from ..misc import make_logger, is_main_process, AverageMeter, get_lr, get_tile_names_from_dataloader, plot_pix2poly
from ..datasets import get_train_loader, get_val_loader
from ..models.pix2poly import EncoderDecoder, ImageEncoder, LiDAREncoder, MultiEncoder, Decoder
from ..predict import Predictor
from ..eval import Evaluator

class NewTrainer:
    def __init__(self, cfg, rank, world_size):
        
        self.cfg = cfg
        logging_level = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger("Trainer",level=logging_level)
        self.logger.info(f"Create output directory {self.cfg.output_dir}")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.lr_scheduler = None
        self.loss_fn_dict = {}
        
        self.is_ddp = self.cfg.multi_gpu
        
        self.update_pbar_every = 1
        
        
    def setup_ddp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)
        
    def init_wandb(self):
    
        if self.cfg.host.name == "jeanzay":
            os.environ["WANDB_MODE"] = "offline"
        
        cfg_container = OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True
        )

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="HiSup",
            name=self.cfg.experiment_name,
            group="v1_pix2poly",
            # track hyperparameters and run metadata
            config=cfg_container,
            dir=self.cfg.output_dir,
        )
        
        log_outfile = os.path.join(self.cfg.output_dir, 'wandb.log')
        wandb.run.log_code(log_outfile)

    def cleanup(self):
        dist.destroy_process_group()

    def setup_tokenizer(self):
        self.tokenizer = Tokenizer(num_classes=1,
            num_bins=self.cfg.model.tokenizer.num_bins,
            width=self.cfg.model.encoder.input_width,
            height=self.cfg.model.encoder.input_height,
            max_len=self.cfg.model.tokenizer.max_len
        )
    
    def compute_dynamic_cfg_vars(self):
    
        self.cfg.model.tokenizer.pad_idx = self.tokenizer.PAD_code
        self.cfg.model.tokenizer.max_len = self.cfg.model.tokenizer.n_vertices*2+2
        self.cfg.model.tokenizer.generation_steps = self.cfg.model.tokenizer.n_vertices*2+1
        self.cfg.model.encoder.num_patches = int((self.cfg.model.encoder.input_size // self.cfg.model.encoder.patch_size) ** 2)
    
    def setup_model(self):
        
        if self.cfg.use_images and self.cfg.use_lidar:
            encoder = MultiEncoder(self.cfg)
        elif self.cfg.use_images:
            encoder = ImageEncoder(self.cfg)
        elif self.cfg.use_lidar: 
            encoder = LiDAREncoder(self.cfg)
        else:
            raise ValueError("At least one of use_image or use_lidar must be True")
        
        decoder = Decoder(
            vocab_size=self.tokenizer.vocab_size,
            encoder_len=self.cfg.model.encoder.num_patches,
            dim=256,
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
            model = DDP(model, device_ids=[self.rank])
        
        self.model = model

    def setup_optimizer(self):
        # Get optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))

        # Get scheduler
        num_training_steps = self.cfg.model.num_epochs * (len(self.train_loader.dataset) // self.cfg.model.batch_size // self.world_size)
        num_warmup_steps = int(0.05 * num_training_steps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )

    def save_checkpoint(self, outfile, **kwargs):
        
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        if self.is_ddp:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            "cfg": self.cfg,
            "model": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, outfile)
        
        self.logger.info(f"Save model {os.path.split(outfile)[-1]} to {outfile}")


    def average_across_gpus(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor
    
    def setup_loss_fn_dict(self):
        
        # Get loss functions
        weight = torch.ones(self.cfg.model.tokenizer.pad_idx + 1, device=self.cfg.device)
        weight[self.tokenizer.num_bins:self.tokenizer.BOS_code] = 0.0
        self.loss_fn_dict["vertex"] = nn.CrossEntropyLoss(ignore_index=self.cfg.model.tokenizer.pad_idx, label_smoothing=self.cfg.model.label_smoothing, weight=weight)
        self.loss_fn_dict["perm"] = nn.BCELoss()
        
        
    def valid_one_epoch(self):
        
        self.logger.info("Validating...")
        self.model.eval()
        self.loss_fn_dict["vertex"].eval()
        self.loss_fn_dict["perm"].eval()

        loss_meter = AverageMeter()
        vertex_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()

        loader = tqdm(self.val_loader, total=len(self.val_loader), file=sys.stdout, dynamic_ncols=True, mininterval=self.update_pbar_every)

        with torch.no_grad():
            for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in loader:
                
                batch_size = x_image.size(0) if self.cfg.use_images else x_lidar.size(0)
                
                if self.cfg.use_images:
                    x_image = x_image.to(self.cfg.device, non_blocking=True)
                if self.cfg.use_lidar:
                    x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)    
                
                y_sequence = y_sequence.to(self.cfg.device, non_blocking=True)
                y_perm = y_perm.to(self.cfg.device, non_blocking=True)

                y_input = y_sequence[:, :-1]
                y_expected = y_sequence[:, 1:]

                preds, perm_mat = self.model(x_image, x_lidar, y_input)


                vertex_loss_weight = self.cfg.model.vertex_loss_weight
                perm_loss_weight = self.cfg.model.perm_loss_weight
                
                vertex_loss = vertex_loss_weight*self.loss_fn_dict["vertex"](preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
                perm_loss = perm_loss_weight*self.loss_fn_dict["perm"](perm_mat, y_perm)

                loss = vertex_loss + perm_loss

                loss_meter.update(loss.item(), batch_size)
                vertex_loss_meter.update(vertex_loss.item(), batch_size)
                perm_loss_meter.update(perm_loss.item(), batch_size)

            loss_dict = {
            'total_loss': loss_meter.avg,
            'vertex_loss': vertex_loss_meter.avg,
            'perm_loss': perm_loss_meter.avg,
        }

        return loss_dict


    def train_one_epoch(self, epoch, iter_idx):
        self.logger.info(f"Train epoch {epoch}...")
        self.model.train()
        self.loss_fn_dict["vertex"].train()
        self.loss_fn_dict["perm"].train()

        loss_meter = AverageMeter()
        vertex_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()

        loader = tqdm(self.train_loader, total=len(self.train_loader), file=sys.stdout, dynamic_ncols=True, mininterval=self.update_pbar_every)

        for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, tile_ids in loader:
            
            batch_size = x_image.size(0) if self.cfg.use_images else x_lidar.size(0)
            
            # ### debug vis
            if self.cfg.debug_vis:
                file_names = get_tile_names_from_dataloader(self.train_loader.dataset.coco.imgs, tile_ids)
                plot_pix2poly(image_batch=x_image,lidar_batch=x_lidar,mask_batch=y_mask,corner_image_batch=y_corner_mask,tile_names=file_names)        
            
            if self.cfg.use_images:
                x_image = x_image.to(self.cfg.device, non_blocking=True)
            if self.cfg.use_lidar:
                x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)
            
            y_sequence = y_sequence.to(self.cfg.device, non_blocking=True)
            y_perm = y_perm.to(self.cfg.device, non_blocking=True)

            y_input = y_sequence[:, :-1]
            y_expected = y_sequence[:, 1:]

            preds, perm_mat = self.model(x_image, x_lidar, y_input)

            if epoch < self.cfg.model.milestone:
                vertex_loss_weight = self.cfg.model.vertex_loss_weight
                perm_loss_weight = 0.0
            else:
                vertex_loss_weight = self.cfg.model.vertex_loss_weight
                perm_loss_weight = self.cfg.model.perm_loss_weight

            vertex_loss = vertex_loss_weight*self.loss_fn_dict["vertex"](preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            perm_loss = perm_loss_weight*self.loss_fn_dict["perm"](perm_mat, y_perm)

            loss = vertex_loss + perm_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step()

            loss_meter.update(loss.item(), batch_size)
            vertex_loss_meter.update(vertex_loss.item(), batch_size)
            perm_loss_meter.update(perm_loss.item(), batch_size)

            lr = get_lr(self.optimizer)

            loader.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")

            iter_idx += 1

            if self.cfg.run_type.name=="debug" and iter_idx % 10 == 0:
                break

        self.logger.info(f"Total train loss: {loss_meter.avg}")
        loss_dict = {
            'total_loss': loss_meter.avg,
            'vertex_loss': vertex_loss_meter.avg,
            'perm_loss': perm_loss_meter.avg,
        }

        return loss_dict, iter_idx



    def train_val_loop(self):

        if self.cfg.log_to_wandb and is_main_process:
            self.init_wandb()

        best_loss = float('inf')

        iter_idx=self.cfg.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.model.start_epoch, self.cfg.model.num_epochs)

        predictor = Predictor(self.cfg)

        if is_main_process():
            evaluator = Evaluator(self.cfg)
            
        for epoch in tqdm(epoch_iterator, position=0, leave=True, file=sys.stdout, dynamic_ncols=True, mininterval=20.0):
            
            ############################################
            ################# Training #################
            ############################################
            
            # important to shuffle the data differently for each epoch, see: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if self.cfg.multi_gpu:
                self.train_loader.sampler.set_epoch(epoch) 

            train_loss_dict, iter_idx = self.train_one_epoch(epoch,iter_idx)
            # Sync all processes before validation
            if self.cfg.multi_gpu:
                dist.barrier()
                
                
            wandb_dict ={}
            wandb_dict['epoch'] = epoch
            for k, v in train_loss_dict.items():
                wandb_dict[f"train_{k}"] = v
            wandb_dict['lr'] = get_lr(self.optimizer)

            ############################################
            ################ Validation ################
            ############################################
            val_loss_dict = self.valid_one_epoch()
            for k, v in val_loss_dict.items():
                wandb_dict[f"val_{k}"] = v
            self.logger.info(f"Valid loss: {val_loss_dict['total_loss']:.3f}\n\n")

            validation_best = False
            # Save best validation loss epoch.
            if val_loss_dict['total_loss'] < best_loss and self.cfg.save_best:
                validation_best = True
                best_loss = val_loss_dict['total_loss']
                checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "validation_best.pth")
                self.save_checkpoint(checkpoint_file, epoch=epoch)

            # Save latest checkpoint every epoch.
            if self.cfg.save_latest:
                checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "latest.pth")
                self.save_checkpoint(checkpoint_file, epoch=epoch)


            if (epoch + 1) % self.cfg.save_every == 0:
                checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"epoch_{epoch}.pth")
                self.save_checkpoint(checkpoint_file, epoch=epoch)

            #############################################
            ############## COCO Evaluation ##############
            #############################################
            if (epoch + 1) % self.cfg.val_every == 0:
                self.logger.info("Predict and evaluate validation set with latest model...")
                coco_predictions = predictor.predict_from_loader(self.model,self.tokenizer,self.val_loader)
                if len(coco_predictions) > 0:
                    self.logger.info(f"Predicted {len(coco_predictions)}/{len(self.val_loader.dataset.coco.getAnnIds())} polygons in the validation set.") 

                    wandb_dict[f"val_num_polygons"] = len(coco_predictions)

                    prediction_outfile = os.path.join(self.cfg.output_dir, "predictions", f"epoch_{epoch}.json")
                    os.makedirs(os.path.dirname(prediction_outfile), exist_ok=True)
                    with open(prediction_outfile, "w") as fp:
                        fp.write(json.dumps(coco_predictions))
                    if validation_best:
                        best_prediction_outfile = os.path.join(self.cfg.output_dir, "predictions", "validation_best.json")
                        shutil.copyfile(prediction_outfile, best_prediction_outfile)
                    
                    evaluator.load_predictions(prediction_outfile)
                    val_metrics_dict = evaluator.evaluate()

                    for metric, value in val_metrics_dict.items():
                        wandb_dict[f"val_{metric}"] = value
                    
                else:
                    self.logger.info("No polygons predicted. Skipping evaluation...")

            self.logger.info("Validation finished...\n")

                
            # Sync all processes before next epoch
            if self.cfg.multi_gpu:
                dist.barrier()

            if self.cfg.log_to_wandb:
                if is_main_process():
                    wandb.log(wandb_dict)

    


    def train(self):
        self.setup_ddp()
        
        self.setup_tokenizer()
        self.compute_dynamic_cfg_vars()
        self.setup_model()
        self.train_loader = get_train_loader(self.cfg,tokenizer=self.tokenizer)
        self.val_loader = get_val_loader(self.cfg,tokenizer=self.tokenizer)
        self.setup_optimizer()
        self.setup_loss_fn_dict()

        self.train_val_loop()
        self.cleanup()


def spawn_worker(rank, world_size, cfg):
    trainer = NewTrainer(cfg, rank, world_size)
    trainer.train()


