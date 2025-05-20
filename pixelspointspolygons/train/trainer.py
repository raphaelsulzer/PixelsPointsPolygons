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
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf

from ..misc import make_logger, get_lr, seed_everything, smart_load_state_dict
from ..datasets import get_train_loader, get_val_loader
from ..predict import Predictor
from ..eval import Evaluator

class Trainer:
    def __init__(self, cfg, local_rank, world_size):
        
        self.cfg = cfg
        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        if verbosity == logging.INFO and local_rank != 0:
            verbosity = logging.WARNING
        self.verbosity = verbosity
        self.logger = make_logger(f"Trainer (rank {local_rank})",level=verbosity)
        self.update_pbar_every = cfg.host.update_pbar_every

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
        
        self.is_ddp = self.cfg.host.multi_gpu
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        
        
    def progress_bar(self,item,start=0):
        
        disable = self.verbosity >= logging.WARNING
        
        pbar = tqdm(item, total=len(item), initial=start,
                      file=sys.stdout, 
                      mininterval=self.update_pbar_every,                      
                      disable=disable,
                      position=0,
                      leave=True)
    
        return pbar
        
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
            name=self.cfg.experiment.name,
            group=self.cfg.experiment.group_name,
            # track hyperparameters and run metadata
            config=cfg_container,
            dir=self.cfg.output_dir,
        )
        
        log_outfile = os.path.join(self.cfg.output_dir, 'wandb.log')
        wandb.run.log_code(log_outfile)

    def average_across_gpus(self, meter):
        
        if not self.is_ddp:
            return meter.global_avg
        
        tensor = torch.tensor([meter.global_avg], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor.item()

    
    def setup_dataloader(self):
        self.train_loader = get_train_loader(self.cfg,logger=self.logger)
        self.val_loader = get_val_loader(self.cfg,logger=self.logger)

    def save_checkpoint(self, outfile, **kwargs):
        """Save checkpoint to file. This is a generic function that saves the model, optimizer and scheduler state dicts."""
        
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        checkpoint = {
            "cfg": self.cfg,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, outfile)
        
        self.logger.info(f"Save model {os.path.split(outfile)[-1]} to {outfile}")

    def save_best_and_latest_checkpoint(self, epoch, val_loss_dict, val_metrics_dict):
        
        # Save best validation loss/iou epoch.
        if val_loss_dict['total_loss'] < self.cfg.training.best_val_loss and self.cfg.training.save_best:
            self.cfg.training.best_val_loss = val_loss_dict['total_loss']
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "best_val_loss.pth")
            self.save_checkpoint(checkpoint_file, epoch=epoch, best_val_loss=self.cfg.training.best_val_loss, best_val_iou=self.cfg.training.best_val_iou)

        if val_metrics_dict.get('IoU',0.0) > self.cfg.training.best_val_iou and self.cfg.training.save_best:

            self.cfg.training.best_val_iou = val_metrics_dict['IoU']
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "best_val_iou.pth")
            self.save_checkpoint(checkpoint_file, epoch=epoch, best_val_loss=self.cfg.training.best_val_loss, best_val_iou=self.cfg.training.best_val_iou)

        # Save latest checkpoint every epoch.
        if self.cfg.training.save_latest:
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "latest.pth")
            self.save_checkpoint(checkpoint_file, epoch=epoch, best_val_loss=self.cfg.training.best_val_loss, best_val_iou=self.cfg.training.best_val_iou)


        if (epoch + 1) % self.cfg.training.save_every == 0:
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"epoch_{epoch}.pth")
            self.save_checkpoint(checkpoint_file, epoch=epoch, best_val_loss=self.cfg.training.best_val_loss, best_val_iou=self.cfg.training.best_val_iou)
    
    

    
    def load_checkpoint(self):
        """Load checkpoint from file. This is a generic function that loads the model, optimizer and scheduler state dicts."""
        
        ## get the file
        if self.cfg.checkpoint_file is not None:
            checkpoint_file = self.cfg.checkpoint_file
            self.cfg.checkpoint = os.path.basename(checkpoint_file).split(".")[0]+"_overwrite"
        else:
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"{self.cfg.checkpoint}.pth")
        if not os.path.isfile(checkpoint_file):
            raise FileExistsError(f"Checkpoint file {checkpoint_file} not found.")
        self.logger.info(f"Loading model from checkpoint: {checkpoint_file}")
        
        ## load the checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.cfg.host.device)
        
        temp = {}
        for k,v in checkpoint.items():
            if "_state_dict" in k:
                temp[k.replace("_state_dict","")] = v
            else:
                temp[k] = v
        checkpoint = temp
        del temp
        
        ## check for correct model type
        cfg = checkpoint.get("cfg",None)
        if cfg is not None:
            if not cfg.experiment.encoder.use_lidar == self.cfg.experiment.encoder.use_lidar:
                self.logger.error(f"Model checkpoint was trained with use_lidar={cfg.experiment.encoder.use_lidar}, but current config is use_lidar={self.cfg.experiment.encoder.use_lidar}.")
                raise ValueError("Model checkpoint and current config do not match.")
            if not cfg.experiment.encoder.use_images == self.cfg.experiment.encoder.use_images:
                self.logger.error(f"Model checkpoint was trained with use_images={cfg.experiment.encoder.use_images}, but current config is use_images={self.cfg.experiment.encoder.use_images}.")
                raise ValueError("Model checkpoint and current config do not match.")
            
            if hasattr(cfg, "model.fusion") and isattr(self.cfg.experiment.model, "fusion"):
                if not cfg.experiment.model.fusion == self.cfg.experiment.model.fusion:
                    self.logger.error(f"Model checkpoint was trained with fusion={cfg.experiment.model.fusion}, but current config is fusion={self.cfg.experiment.model.fusion}.")
                    raise ValueError("Model checkpoint and current config do not match.")   
        
        self.model = smart_load_state_dict(self.model, checkpoint["model"], self.logger, strict=True)

        self.lr_scheduler.load_state_dict(checkpoint.get("scheduler",checkpoint.get("lr_scheduler")))
        if "loss_func" in checkpoint:
            self.loss_func.load_state_dict(checkpoint["loss_func"])
        
        start_epoch = checkpoint.get("epochs_run",checkpoint.get("epoch",0))
        self.cfg.experiment.model.start_epoch = start_epoch + 1
        
        self.cfg.training.best_val_loss = checkpoint.get("best_val_loss",self.cfg.training.best_val_loss)
        self.cfg.training.best_val_iou = checkpoint.get("best_val_iou",self.cfg.training.best_val_iou)


    def train_val_loop(self):
        """This is just an example of how to use the Trainer class. The actual train_val_loop slighlty varies for different architectures."""

        if self.cfg.checkpoint is not None:
            best_val_loss, best_val_iou = self.load_checkpoint()
        else:
            best_val_loss = np.float('inf')
            best_val_iou = np.float('-inf')
            
        if self.cfg.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()

        iter_idx=self.cfg.experiment.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.experiment.model.start_epoch, self.cfg.experiment.model.num_epochs)

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
                if val_loss_dict['total_loss'] < best_loss and self.cfg.training.save_best:
                    validation_best = True
                    best_loss = val_loss_dict['total_loss']
                    checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "validation_best.pth")
                    self.save_checkpoint(checkpoint_file, epoch=epoch)

                # Save latest checkpoint every epoch.
                if self.cfg.training.save_latest:
                    checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", "latest.pth")
                    self.save_checkpoint(checkpoint_file, epoch=epoch)


                if (epoch + 1) % self.cfg.training.save_every == 0:
                    checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"epoch_{epoch}.pth")
                    self.save_checkpoint(checkpoint_file, epoch=epoch)

            #############################################
            ############## COCO Evaluation ##############
            #############################################
            if (epoch + 1) % self.cfg.training.val_every == 0:

                self.logger.info("Predict validation set with latest model...")
                coco_predictions = predictor.predict_from_loader(self.model,self.tokenizer,self.val_loader)
                
                
                print(f"rank {self.local_rank}, device: {self.device}, coco_pred_type: {type(coco_predictions)}, coco_pred_len: {len(coco_predictions)}")
                
                if self.is_ddp:
                    
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

    def setup_model(self):
        pass
    def setup_optimizer(self):
        pass
    def setup_loss_fn_dict(self):
        pass
    
    def cleanup(self):
        dist.destroy_process_group()
        
    def train(self):
        seed_everything(42)
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_loss_fn_dict()
        self.train_val_loop()
        self.cleanup()