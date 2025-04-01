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

from ..models.pix2poly import Tokenizer, ImageEncoder, LiDAREncoder, MultiEncoder, EncoderDecoder, Decoder
from ..misc import AverageMeter, make_logger, get_lr, get_tile_names_from_dataloader, plot_pix2poly, seed_everything
from ..datasets import get_train_loader, get_val_loader
from ..predict.predictor_pix2poly import Pix2PolyPredictor as Predictor
from ..eval import Evaluator
from .trainer import Trainer

class Pix2PolyTrainer(Trainer):
    def __init__(self, cfg, local_rank, world_size):
        
        self.cfg = cfg
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        if verbosity == logging.INFO and local_rank != 0:
            verbosity = logging.WARNING
        self.verbosity = verbosity
        self.logger = make_logger(f"Trainer (rank {local_rank})",level=verbosity)
        self.update_pbar_every = cfg.update_pbar_every

        self.logger.log(logging.INFO, f"Init Trainer on rank {local_rank} in world size {world_size}...")
        self.logger.info("Configuration:")
        self.logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
        if local_rank == 0:
            self.logger.info(f"Create output directory {self.cfg.output_dir}")
            os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.lr_scheduler = None
        self.loss_fn_dict = {}
        
        self.is_ddp = self.cfg.multi_gpu
        
        
    def progress_bar(self,item):
        
        disable = self.verbosity >= logging.WARNING
        
        pbar = tqdm(item, total=len(item), 
                      file=sys.stdout, 
                      mininterval=self.update_pbar_every,                      
                      disable=disable,
                      position=0,
                      leave=True)
    
        return pbar
    
    def setup_ddp(self):

        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
        dist_url = "env://"  # default

        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=self.world_size,
            rank=int(os.environ["RANK"])
        )
        
        # this will make all .cuda() calls work properly.
        torch.cuda.set_device(self.local_rank)

        # synchronizes all threads to reach this point before moving on.
        dist.barrier()
        
    def setup_wandb(self):
    
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


    def setup_optimizer(self):
        # Get optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))

        # Get scheduler
        # num_training_steps = self.cfg.model.num_epochs * (len(self.train_loader.dataset) // self.cfg.model.batch_size // self.world_size)                
        num_training_steps = self.cfg.model.num_epochs * len(self.train_loader)
        self.logger.debug(f"Number of training steps on this GPU: {num_training_steps}")
        self.logger.info(f"Total number of training steps: {num_training_steps*self.world_size}")
        
        num_warmup_steps = int(0.05 * num_training_steps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
        
    def setup_tokenizer(self):
        self.tokenizer = Tokenizer(num_classes=1,
            num_bins=self.cfg.model.tokenizer.num_bins,
            width=self.cfg.model.encoder.input_width,
            height=self.cfg.model.encoder.input_height,
            max_len=self.cfg.model.tokenizer.max_len
        )
    
    def setup_cfg_vars(self):
    
        self.cfg.model.tokenizer.pad_idx = self.tokenizer.PAD_code
        self.cfg.model.tokenizer.max_len = self.cfg.model.tokenizer.n_vertices*2+2
        self.cfg.model.tokenizer.generation_steps = self.cfg.model.tokenizer.n_vertices*2+1
        self.cfg.model.encoder.num_patches = int((self.cfg.model.encoder.input_size // self.cfg.model.encoder.patch_size) ** 2)
    
    def setup_model(self):
        
        ## TODO: maybe it is better to make a model class that goes at the beginning of this file, which can then also be used by the Predictor class
        
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
            dim=self.cfg.model.encoder.out_dim,
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
        
        if self.is_ddp:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.local_rank])
        
        self.model = model

    def setup_dataloader(self):
        self.train_loader = get_train_loader(self.cfg,tokenizer=self.tokenizer)
        self.val_loader = get_val_loader(self.cfg,tokenizer=self.tokenizer)

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
            
        checkpoint = {
            "cfg": self.cfg,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, outfile)
        
        self.logger.info(f"Save model {os.path.split(outfile)[-1]} to {outfile}")

    def load_checkpoint(self):
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"{self.cfg.checkpoint}.pth")
        if not os.path.isfile(checkpoint_file):
            raise FileExistsError(f"Checkpoint file {checkpoint_file} does not exist.")

        self.logger.info(f"Load checkpoint {self.cfg.checkpoint} from {checkpoint_file}...")
        
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        self.model.load_state_dict(checkpoint.get("model",checkpoint.get("state_dict")))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        
        start_epoch = checkpoint.get("epochs_run",checkpoint.get("epoch",0))
        self.cfg.model.start_epoch = start_epoch + 1

    def average_across_gpus(self, meter):
        
        if not self.is_ddp:
            return meter.avg
        
        tensor = torch.tensor([meter.avg], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor.item()
    
    
    def setup_loss_fn_dict(self):
        
        # Get loss functions
        weight = torch.ones(self.cfg.model.tokenizer.pad_idx + 1, device=self.cfg.device)
        weight[self.tokenizer.num_bins:self.tokenizer.BOS_code] = 0.0
        self.loss_fn_dict["vertex"] = nn.CrossEntropyLoss(ignore_index=self.cfg.model.tokenizer.pad_idx, label_smoothing=self.cfg.model.label_smoothing, weight=weight)
        self.loss_fn_dict["perm"] = nn.BCELoss()
        
        
    def valid_one_epoch(self):

        self.logger.info("Validate...")
        self.model.eval()
        self.loss_fn_dict["vertex"].eval()
        self.loss_fn_dict["perm"].eval()

        loss_meter = AverageMeter()
        vertex_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()

        loader = self.progress_bar(self.val_loader)

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


        self.logger.debug(f"Validation loss: {loss_meter.avg:.3f}")
        loss_dict = {
            'total_loss': self.average_across_gpus(loss_meter),
            'vertex_loss': self.average_across_gpus(vertex_loss_meter),
            'perm_loss': self.average_across_gpus(perm_loss_meter),
        }
        self.logger.info(f"Validation loss: {loss_dict['total_loss']:.3f}")

        return loss_dict


    def train_one_epoch(self, epoch, iter_idx):
        
        self.logger.info(f"Train epoch {epoch}...")
        
        self.model.train()
        self.loss_fn_dict["vertex"].train()
        self.loss_fn_dict["perm"].train()

        loss_meter = AverageMeter()
        vertex_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()

        
        loader = self.progress_bar(self.train_loader)

        for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, tile_ids in loader:
                        
            batch_size = x_image.size(0) if self.cfg.use_images else x_lidar.size(0)
            
            # ### debug vis
            if self.cfg.debug_vis:
                file_names = get_tile_names_from_dataloader(self.train_loader.dataset.coco.imgs, tile_ids)
                # plot_pix2poly(image_batch=x_image,lidar_batch=x_lidar,mask_batch=y_mask,corner_image_batch=y_corner_mask,tile_names=file_names)        
                plot_pix2poly(image_batch=x_image,lidar_batch=x_lidar,corner_image_batch=y_corner_mask,tile_names=file_names)        
            
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

            # if self.cfg.run_type.name=="debug" and iter_idx % 10 == 0:
            #     break
            
        
        self.logger.debug(f"Train loss: {loss_meter.avg:.3f}")
        loss_dict = {
            'total_loss': self.average_across_gpus(loss_meter),
            'vertex_loss': self.average_across_gpus(vertex_loss_meter),
            'perm_loss': self.average_across_gpus(perm_loss_meter),
        }
        
        self.logger.info(f"Train loss: {loss_dict['total_loss']:.3f}")

        return loss_dict, iter_idx



    def train_val_loop(self):

        if self.cfg.checkpoint is not None:
            self.load_checkpoint()
            
        if self.cfg.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()

        best_loss = float('inf')

        iter_idx=self.cfg.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.model.start_epoch, self.cfg.model.num_epochs)

        predictor = Predictor(self.cfg,local_rank=self.local_rank,world_size=self.world_size)

        if self.local_rank == 0:
            evaluator = Evaluator(self.cfg)
        else:
            evaluator = None
        
        for epoch in self.progress_bar(epoch_iterator):
            
            ############################################
            ################# Training #################
            ############################################
            # important to shuffle the data differently for each epoch
            # see: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch) 

            train_loss_dict, iter_idx = self.train_one_epoch(epoch,iter_idx)
            # Sync all processes before validation
            if self.is_ddp:
                dist.barrier()
            
            if self.local_rank == 0:
                wandb_dict ={}
                wandb_dict['epoch'] = epoch
                for k, v in train_loss_dict.items():
                    wandb_dict[f"train_{k}"] = v
                wandb_dict['lr'] = get_lr(self.optimizer)


            ############################################
            ################ Validation ################
            ############################################
            val_loss_dict = self.valid_one_epoch()
            if self.local_rank == 0:
                for k, v in val_loss_dict.items():
                    wandb_dict[f"val_{k}"] = v

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

                self.logger.info("Predict validation set with latest model...")
                coco_predictions = predictor.predict_from_loader(self.model,self.tokenizer,self.val_loader)
                
                
                self.logger.debug(f"rank {self.local_rank}, device: {self.device}, coco_pred_type: {type(coco_predictions)}, coco_pred_len: {len(coco_predictions)}")
                
                if self.cfg.multi_gpu:
                    
                    # Gather the list of dictionaries from all ranks
                    gathered_predictions = [None] * self.world_size  # Placeholder for gathered objects
                    dist.all_gather_object(gathered_predictions, coco_predictions)

                    # Flatten the list of lists into a single list
                    coco_predictions = [item for sublist in gathered_predictions for item in sublist]
                    
                
                if not len(coco_predictions):
                    self.logger.info("No polygons predicted. Skipping coco evaluation...")
                    continue
                    
                if self.local_rank == 0:
                    self.logger.info(f"Predicted {len(coco_predictions)}/{len(self.val_loader.dataset.coco.getAnnIds())} polygons...") 
                    self.logger.info(f"Run coco evaluation on rank {self.local_rank}...")

                    wandb_dict[f"val_num_polygons"] = len(coco_predictions)

                    prediction_outfile = os.path.join(self.cfg.output_dir, "predictions", f"epoch_{epoch}.json")
                    os.makedirs(os.path.dirname(prediction_outfile), exist_ok=True)
                    with open(prediction_outfile, "w") as fp:
                        fp.write(json.dumps(coco_predictions))
                    self.logger.info(f"Saved predictions to {prediction_outfile}")
                    if validation_best:
                        best_prediction_outfile = os.path.join(self.cfg.output_dir, "predictions", "validation_best.json")
                        shutil.copyfile(prediction_outfile, best_prediction_outfile)
                        self.logger.info(f"Copied predictions to {best_prediction_outfile}")

                    evaluator.load_predictions(prediction_outfile)
                    val_metrics_dict = evaluator.evaluate()

                    for metric, value in val_metrics_dict.items():
                        wandb_dict[f"val_{metric}"] = value
                    
                else:
                    self.logger.info("Rank {self.rank} waiting until coco evaluation is done...")

            self.logger.info("Validation finished...\n")
                
            # Sync all processes before next epoch
            if self.is_ddp:
                dist.barrier()

            if self.cfg.log_to_wandb:
                if self.local_rank == 0:
                    wandb.log(wandb_dict)

    


    def train(self):
        seed_everything(42)
        if self.is_ddp:
            self.setup_ddp()
        self.setup_tokenizer()
        self.setup_cfg_vars()
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_loss_fn_dict()
        self.train_val_loop()
        self.cleanup()