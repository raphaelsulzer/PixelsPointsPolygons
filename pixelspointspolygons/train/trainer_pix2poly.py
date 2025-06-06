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

from collections import defaultdict
from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup

from ..datasets import get_train_loader, get_val_loader
from ..models.pix2poly import Tokenizer, Pix2PolyModel
from ..misc import AverageMeter, get_lr, get_tile_names_from_dataloader, denormalize_image_for_visualization
from ..predict.predictor_pix2poly import Pix2PolyPredictor as Predictor
from ..eval import Evaluator
from ..misc.debug_visualisations import *
from ..misc.coco_conversions import coco_anns_to_shapely_polys, tensor_to_shapely_polys

from .trainer import Trainer

class Pix2PolyTrainer(Trainer):

    def setup_optimizer(self):
        # Get optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.experiment.model.learning_rate, weight_decay=self.cfg.experiment.model.weight_decay, betas=(0.9, 0.95))

        # Get scheduler
        # num_training_steps = self.cfg.experiment.model.num_epochs * (len(self.train_loader.dataset) // self.cfg.experiment.model.batch_size // self.world_size)                
        num_training_steps = self.cfg.experiment.model.num_epochs * len(self.train_loader)
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
            num_bins=self.cfg.experiment.model.tokenizer.num_bins,
            width=self.cfg.experiment.encoder.in_width,
            height=self.cfg.experiment.encoder.in_height,
            max_len=self.cfg.experiment.model.tokenizer.max_len
        )
        self.cfg.experiment.model.tokenizer.pad_idx = self.tokenizer.PAD_code
        self.cfg.experiment.model.tokenizer.max_len = self.cfg.experiment.model.tokenizer.n_vertices*2+2
        self.cfg.experiment.model.tokenizer.generation_steps = self.cfg.experiment.model.tokenizer.n_vertices*2+1
    
    def setup_model(self):
        
        self.setup_tokenizer()
        self.model = Pix2PolyModel(self.cfg,self.tokenizer.vocab_size,local_rank=self.local_rank)
        
    def setup_dataloader(self):
        """Pix2Poly needs a tokenizer in the dataset __get_item__ method to tokenize the polygons. Thus overwrite the setup_dataloader method here."""
        
        self.train_loader = get_train_loader(self.cfg,logger=self.logger,tokenizer=self.tokenizer)
        self.val_loader = get_val_loader(self.cfg,logger=self.logger,tokenizer=self.tokenizer)

    def setup_optimizer(self):
        # Get optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.experiment.model.learning_rate, weight_decay=self.cfg.experiment.model.weight_decay, betas=(0.9, 0.95))

        # Get scheduler
        num_training_steps = self.cfg.experiment.model.num_epochs * (len(self.train_loader.dataset) // self.cfg.experiment.model.batch_size // self.world_size)
        num_warmup_steps = int(0.05 * num_training_steps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
    
    def setup_loss_fn_dict(self):
        
        # Get loss functions
        weight = torch.ones(self.cfg.experiment.model.tokenizer.pad_idx + 1, device=self.cfg.host.device)
        weight[self.tokenizer.num_bins:self.tokenizer.BOS_code] = 0.0
        self.loss_fn_dict["vertex"] = nn.CrossEntropyLoss(ignore_index=self.cfg.experiment.model.tokenizer.pad_idx, label_smoothing=self.cfg.experiment.model.label_smoothing, weight=weight)
        self.loss_fn_dict["perm"] = nn.BCELoss()
    
    
    def visualization(self,  loader, epoch, predictor=None, coco=None, num_images=2):
        
        self.model.eval()
        
        x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, tile_ids = next(iter(loader))
        
        # TODO: maybe plot y_sequence instead of y_corner_mask, because it is the input to the model
        if self.cfg.experiment.encoder.use_images:
            x_image = x_image.to(self.cfg.host.device, non_blocking=True)
            x_image = x_image[:num_images]
        if self.cfg.experiment.encoder.use_lidar:
            x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)
            x_lidar = x_lidar.unbind()
            x_lidar = list(x_lidar)[:num_images]
            x_lidar = torch.nested.nested_tensor(x_lidar, layout=torch.jagged)
        
        outpath = os.path.join(self.cfg.output_dir, "visualizations", f"{epoch}")
        os.makedirs(outpath, exist_ok=True)
        self.logger.info(f"Save visualizations to {outpath}")
        
        if predictor is not None:
            batch_polygons = predictor.batch_to_polygons(x_image, x_lidar, self.model, self.tokenizer)
        coco_anns = defaultdict(list)
        if coco is not None:
            for ann in coco:
                if ann["image_id"] >= num_images: 
                    break
                coco_anns[ann["image_id"]].append(ann)
                
        if self.cfg.experiment.encoder.use_lidar:
            lidar_batches = torch.unbind(x_lidar, dim=0)
            
        names = get_tile_names_from_dataloader(loader, tile_ids.cpu().numpy().flatten().tolist())

        for i in range(num_images):
        
            fig, ax = plt.subplots(1,2,figsize=(8, 4), dpi=150)
            ax = ax.flatten()

            if self.cfg.experiment.encoder.use_images:
                image = denormalize_image_for_visualization(x_image[i], self.cfg)
                plot_image(image, ax=ax[0])
                plot_image(image, ax=ax[1])
            if self.cfg.experiment.encoder.use_lidar:
                plot_point_cloud(lidar_batches[i], ax=ax[0])
                plot_point_cloud(lidar_batches[i], ax=ax[1])
                
            mask_color = [1,0,1,0.6]
            plot_mask(y_mask[i], ax=ax[0], color=mask_color)
            plot_point_activations(y_corner_mask[i], ax=ax[0], color=[1,1,0,1.0])
            
            if coco is not None:
                polygons = coco_anns_to_shapely_polys(coco_anns[i])
            elif predictor is not None:
                polygons = tensor_to_shapely_polys(batch_polygons[i])
            else:
                polygons = []

            if len(polygons):
                plot_shapely_polygons(polygons, ax=ax[1],pointcolor=[1,1,0],edgecolor=[1,0,1])
                
            ax[0].set_title("GT_"+names[i])
            ax[1].set_title("PRED_"+names[i])
            
            plt.tight_layout()
            outfile = os.path.join(outpath, f"{names[i]}.png")
            self.logger.debug(f"Save visualization to {outfile}")
            plt.savefig(outfile)
            if self.cfg.run_type.log_to_wandb and self.local_rank == 0:
                wandb.log({f"{epoch}: {names[i]}": wandb.Image(fig)})            
            plt.close(fig)
        
    def valid_one_epoch(self):

        self.logger.info("Validate...")
        self.model.eval()
        self.loss_fn_dict["vertex"].eval()
        self.loss_fn_dict["perm"].eval()

        loss_meter = AverageMeter()
        vertex_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()

        loader = self.progress_bar(self.val_loader)

        n_points = 0
        
        for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in loader:
            
            batch_size = x_image.size(0) if self.cfg.experiment.encoder.use_images else x_lidar.size(0)
            
            if self.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self.cfg.host.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)    
            
            y_sequence = y_sequence.to(self.cfg.host.device, non_blocking=True)
            y_perm = y_perm.to(self.cfg.host.device, non_blocking=True)

            y_input = y_sequence[:, :-1]
            y_expected = y_sequence[:, 1:]

            preds, perm_mat = self.model(x_image, x_lidar, y_input)


            vertex_loss_weight = self.cfg.experiment.model.vertex_loss_weight
            perm_loss_weight = self.cfg.experiment.model.perm_loss_weight
            
            vertex_loss = vertex_loss_weight*self.loss_fn_dict["vertex"](preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            perm_loss = perm_loss_weight*self.loss_fn_dict["perm"](perm_mat, y_perm)

            loss = vertex_loss + perm_loss

            loss_meter.update(loss.item(), batch_size)
            vertex_loss_meter.update(vertex_loss.item(), batch_size)
            perm_loss_meter.update(perm_loss.item(), batch_size)
            
            x_lidar = x_lidar.unbind()
            x_lidar = list(x_lidar)
            for tensor in x_lidar:
                n_points += tensor.shape[0]
                    
        # n_images = len(self.val_loader.dataset)
        # area = n_images * 3136
        # self.logger.debug(f"Validation pts/m2 {n_points/area}")

        self.logger.debug(f"Validation loss: {loss_meter.global_avg:.3f}")
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
                        
            batch_size = x_image.size(0) if self.cfg.experiment.encoder.use_images else x_lidar.size(0)     
            
            if self.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self.cfg.host.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)
            
            y_sequence = y_sequence.to(self.cfg.host.device, non_blocking=True)
            y_perm = y_perm.to(self.cfg.host.device, non_blocking=True)

            y_input = y_sequence[:, :-1]
            y_expected = y_sequence[:, 1:]

            preds, perm_mat = self.model(x_image, x_lidar, y_input)

            if epoch < self.cfg.experiment.model.milestone:
                vertex_loss_weight = self.cfg.experiment.model.vertex_loss_weight
                perm_loss_weight = 0.0
            else:
                vertex_loss_weight = self.cfg.experiment.model.vertex_loss_weight
                perm_loss_weight = self.cfg.experiment.model.perm_loss_weight

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

            loader.set_postfix(train_loss=loss_meter.global_avg, lr=f"{lr:.5f}")

            iter_idx += 1

            # if self.cfg.run_type.name=="debug" and iter_idx % 10 == 0:
            #     break
            
        
        self.logger.debug(f"Train loss: {loss_meter.global_avg:.3f}")
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
            
        if self.cfg.run_type.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()

        iter_idx=self.cfg.experiment.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.experiment.model.start_epoch, self.cfg.experiment.model.num_epochs)

        predictor = Predictor(self.cfg,local_rank=self.local_rank,world_size=self.world_size)

        if self.local_rank == 0:
            evaluator = Evaluator(self.cfg)
            evaluator.load_gt(self.cfg.dataset.annotations["val"])
        else:
            evaluator = None
        
        for epoch in self.progress_bar(epoch_iterator, start=self.cfg.experiment.model.start_epoch):
            
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
            
            with torch.no_grad():
                if self.local_rank == 0:
                    self.visualization(self.train_loader,epoch,predictor=predictor)
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

                #############################################
                ############## COCO Evaluation ##############
                #############################################
                val_metrics_dict = {}
                if (epoch + 1) % self.cfg.training.val_every == 0:

                    self.logger.info("Predict validation set with latest model...")
                    coco_predictions = predictor.predict_from_loader(self.model,self.tokenizer,self.val_loader)
                    
                    
                    self.logger.debug(f"rank {self.local_rank}, device: {self.device}, coco_pred_type: {type(coco_predictions)}, coco_pred_len: {len(coco_predictions)}")
                    
                    if self.cfg.host.multi_gpu:
                        
                        # Gather the list of dictionaries from all ranks
                        gathered_predictions = [None] * self.world_size  # Placeholder for gathered objects
                        dist.all_gather_object(gathered_predictions, coco_predictions)

                        # Flatten the list of lists into a single list
                        coco_predictions = [item for sublist in gathered_predictions for item in sublist]
                    
                    if not len(coco_predictions):
                        self.logger.info("No polygons predicted. Skipping coco evaluation...")
                    else:
                        self.visualization(self.val_loader,epoch,coco=coco_predictions)
                    
                    if self.local_rank == 0 and len(coco_predictions):
                        self.logger.info(f"Predicted {len(coco_predictions)}/{len(self.val_loader.dataset.coco.getAnnIds())} polygons...") 
                        self.logger.info(f"Run coco evaluation on rank {self.local_rank}...")

                        wandb_dict[f"val_num_polygons"] = len(coco_predictions)

                        prediction_outfile = os.path.join(self.cfg.output_dir, f"predictions_{self.cfg.experiment.country}_{self.cfg.evaluation.split}", f"epoch_{epoch}.json")
                        os.makedirs(os.path.dirname(prediction_outfile), exist_ok=True)
                        with open(prediction_outfile, "w") as fp:
                            fp.write(json.dumps(coco_predictions))
                        self.logger.info(f"Saved predictions to {prediction_outfile}")                        

                        evaluator.load_predictions(prediction_outfile)
                        val_metrics_dict = evaluator.evaluate()
                        evaluator.print_dict_results(val_metrics_dict)
                        
                        if val_metrics_dict['IoU'] > self.cfg.training.best_val_iou:
                            best_prediction_outfile = os.path.join(self.cfg.output_dir, f"predictions_{self.cfg.experiment.country}_{self.cfg.evaluation.split}", "best_val_iou.json")
                            shutil.copyfile(prediction_outfile, best_prediction_outfile)
                            self.logger.info(f"Copied predictions to {best_prediction_outfile}")

                        for metric, value in val_metrics_dict.items():
                            wandb_dict[f"val_{metric}"] = value
                
                
                if self.local_rank == 0:
                    self.save_best_and_latest_checkpoint(epoch, val_loss_dict, val_metrics_dict)
                    for k,v in wandb_dict.items():
                        self.logger.debug(f"{k}: {v}")
                        if self.cfg.run_type.log_to_wandb:
                            wandb.log(wandb_dict)
                            
                # Sync all processes before next epoch
                if self.cfg.host.multi_gpu:
                    dist.barrier()

        

