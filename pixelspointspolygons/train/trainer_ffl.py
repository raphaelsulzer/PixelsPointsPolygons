# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import os
import torch
import wandb
import json
import shutil
import torch

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import optim
from transformers import  get_cosine_schedule_with_warmup
from torchvision.models.segmentation._utils import _SimpleSegmentationModel

from ..misc import get_lr, get_tile_names_from_dataloader, plot_hisup, seed_everything, MetricLogger
from ..models.ffl import *
from ..models.ffl.losses import build_combined_loss
from ..models.ffl.local_utils import batch_to_cuda
from ..models.ffl.measures import iou as compute_iou
from ..predict.ffl.predictor_ffl import FFLPredictor as Predictor
from ..eval import Evaluator

from .trainer import Trainer

class FFLTrainer(Trainer):
    
    def to_device(self,data,device):
        if isinstance(data,torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
    #         import pdb; pdb.set_trace()
            for key in data:
                if isinstance(data[key],torch.Tensor):
                    data[key] = data[key].to(device)
            return data
        if isinstance(data,list):
            return [self.to_device(d,device) for d in data]

    def to_single_device(self,data,device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            return data
        if isinstance(data, list):
            return [self.to_device(d, device) for d in data]
    
    def setup_model(self):
        
        if self.cfg.use_images and self.cfg.use_lidar:
            model = MultiEncoderDecoder(self.cfg)
        elif self.cfg.use_images:
            encoder = UNetResNetBackbone(self.cfg)
            encoder = _SimpleSegmentationModel(encoder, classifier=torch.nn.Identity())
        elif self.cfg.use_lidar: 
            model = LiDAREncoderDecoder(self.cfg)
        else:
            raise ValueError("At least one of use_image or use_lidar must be True")
        
        model = EncoderDecoder(
            encoder=encoder,
            cfg=self.cfg
        )
                
        model.to(self.cfg.device)
        
        if self.is_ddp:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.local_rank])
        
        self.model = model

        
    def setup_optimizer(self):
        # Get optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))

        # Get scheduler
        # num_training_steps = self.cfg.model.num_epochs * (len(self.train_loader.dataset) // self.cfg.model.batch_size // self.world_size)
        num_training_steps = self.cfg.model.num_epochs * len(self.train_loader)
        self.logger.debug(f"Number of training steps on this GPU: {num_training_steps}")
        self.logger.info(f"Total number of training steps: {num_training_steps*self.world_size}")
        # num_warmup_steps = int(0.05 * num_training_steps)
        num_warmup_steps = 0
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
    
    def setup_loss_fn_dict(self):
        
        self.loss_func = build_combined_loss(self.cfg).cuda()
        
    def average_across_gpus(self, meter):
        
        if not self.is_ddp:
            return meter.global_avg
        
        tensor = torch.tensor([meter.global_avg], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor.item()

    def compute_loss_norms(self, dl, total_batches):
        self.loss_func.reset_norm()
        
        self.logger.info("Init loss norms...")
        t = self.progress_bar(dl)  # Initialise

        batch_i = 0
        while batch_i < total_batches:
            for batch in dl:
                # Update loss norms
                batch = batch_to_cuda(batch)
                pred, batch = self.model(batch)
                self.loss_func.update_norm(pred, batch, batch["image"].shape[0])
                if t is not None:
                    t.update(1)
                batch_i += 1
                if not batch_i < total_batches:
                    break

        # Now sync loss norms across GPUs:
        if self.cfg.multi_gpu:
            self.loss_func.sync(self.world_size)
        
    def valid_one_epoch(self, epoch):

        self.logger.info("Validate...")
        
        self.model.eval()

        loss_meter = MetricLogger(" val_")

        loader = self.progress_bar(self.val_loader)

        
        with torch.no_grad():
            for batch_dict in loader:
                
                batch_dict = batch_to_cuda(batch_dict)
                
                pred, batch = self.model(batch_dict)
                loss, loss_dict, extra_dict = self.loss_func(pred, batch, epoch=epoch)
            
                with torch.no_grad():
                    # Compute IoUs at different thresholds
                    if "seg" in pred:
                        y_pred = pred["seg"][:, 0, ...]
                        y_true = batch["gt_polygons_image"][:, 0, ...]
                        iou_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
                        for iou_threshold in iou_thresholds:
                            iou = compute_iou(y_pred.reshape(y_pred.shape[0], -1), y_true.reshape(y_true.shape[0], -1), threshold=iou_threshold)
                            mean_iou = torch.mean(iou)
                            loss_dict[f"IoU_{iou_threshold}"] = mean_iou
                        
                    loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                    loss_reduced = loss.item()
                    loss_meter.update(total_loss=loss_reduced, **loss_dict_reduced)

        self.logger.debug(f"Validation loss: {loss_meter.meters['total_loss'].global_avg:.3f}")
        
        for k,v in loss_meter.meters.items():
            loss_dict[k] = self.average_across_gpus(v)
        
        self.logger.info(f"Validation loss: {loss_dict['total_loss']:.3f}")

        return loss_dict



    def train_one_epoch(self, epoch, iter_idx):
        
        self.logger.info(f"Train epoch {epoch}...")
        
        self.model.train()

        loss_meter = MetricLogger(" train_")

        loader = self.progress_bar(self.train_loader)

        self.loss_func.reset_norm()
        for batch_dict in loader:
                        
            # ### debug vis
            if self.cfg.debug_vis:
                file_names = get_tile_names_from_dataloader(self.train_loader.dataset.coco.imgs, tile_ids)
                plot_hisup(image_batch=x_image,lidar_batch=x_lidar,annotations_batch=y,tile_names=file_names)

            batch_dict = batch_to_cuda(batch_dict)
            
            pred, batch = self.model(batch_dict)
            loss, loss_dict, extra_dict = self.loss_func(pred, batch, epoch=epoch)
                
            with torch.no_grad():
                # Compute IoUs at different thresholds
                if "seg" in pred:
                    y_pred = pred["seg"][:, 0, ...]
                    y_true = batch["gt_polygons_image"][:, 0, ...]
                    iou_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
                    for iou_threshold in iou_thresholds:
                        iou = compute_iou(y_pred.reshape(y_pred.shape[0], -1), y_true.reshape(y_true.shape[0], -1), threshold=iou_threshold)
                        mean_iou = torch.mean(iou)
                        loss_dict[f"IoU_{iou_threshold}"] = mean_iou
                
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = loss.item()
                loss_meter.update(total_loss=loss_reduced, **loss_dict_reduced)
            

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step()

            lr = get_lr(self.optimizer)

            loader.set_postfix(train_loss=loss_meter.meters["total_loss"].global_avg, lr=f"{lr:.5f}")

            iter_idx += 1

            
        
        self.logger.debug(f"Train loss: {loss_meter.meters['total_loss'].global_avg:.3f}")
        
        for k,v in loss_meter.meters.items():
            loss_dict[k] = self.average_across_gpus(v)
        
        self.logger.info(f"Train loss: {loss_dict['total_loss']:.3f}")

        return loss_dict, iter_idx


    def train_val_loop(self):

        if self.cfg.checkpoint is not None:
            self.load_checkpoint()
            
        if self.cfg.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()

        predictor = Predictor(self.cfg,local_rank=self.local_rank,world_size=self.world_size)

        ## init loss norms
        self.model.train()  # Important for batchnorm and dropout, even in computing loss norms
        with torch.no_grad():
            loss_norm_batches_min = self.cfg.model.loss.multiloss.normalization_params.min_samples // (2 * self.cfg.model.batch_size) + 1
            loss_norm_batches_max = self.cfg.model.loss.multiloss.normalization_params.max_samples // (2 * self.cfg.model.batch_size) + 1
            loss_norm_batches = max(loss_norm_batches_min, min(loss_norm_batches_max, len(self.train_loader)))
            self.compute_loss_norms(self.train_loader, loss_norm_batches)

        best_loss = float('inf')

        iter_idx=self.cfg.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.model.start_epoch, self.cfg.model.num_epochs)

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
            val_loss_dict = self.valid_one_epoch(epoch)
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
                for k,v in wandb_dict.items():
                    self.logger.debug(f"{k}: {v}")
                if self.local_rank == 0:
                    wandb.log(wandb_dict)

    


    def train(self):
        seed_everything(42)
        if self.is_ddp:
            self.setup_ddp()
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_loss_fn_dict()
        self.train_val_loop()
        self.cleanup()