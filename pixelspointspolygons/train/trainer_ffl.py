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

from collections import defaultdict
from transformers import  get_cosine_schedule_with_warmup

from ..misc import get_lr, MetricLogger, get_tile_names_from_dataloader, denormalize_image_for_visualization
from ..models.ffl.losses import build_combined_loss
from ..models.ffl.local_utils import batch_to_cuda
from ..models.ffl.model_ffl import FFLModel
from ..predict.predictor_ffl import FFLPredictor as Predictor
from ..eval import Evaluator

from ..misc.coco_conversions import coco_anns_to_shapely_polys
from ..misc.debug_visualisations import *
from lydorn_utils.math_utils import compute_crossfield_uv
import numpy as np

from .trainer import Trainer

class FFLTrainer(Trainer):
    
    def setup_model(self):
        self.model = FFLModel(self.cfg, self.local_rank)
        
    def setup_optimizer(self):
        # Get optimizer
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.experiment.model.learning_rate, weight_decay=self.cfg.experiment.model.weight_decay, betas=(0.9, 0.95))

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.cfg.experiment.model.learning_rate,eps=1e-8)
        # Get scheduler
        # num_training_steps = self.cfg.experiment.model.num_epochs * (len(self.train_loader.dataset) // self.cfg.experiment.model.batch_size // self.world_size)
        num_training_steps = self.cfg.experiment.model.num_epochs * len(self.train_loader)
        self.logger.debug(f"Number of training steps on this GPU: {num_training_steps}")
        self.logger.info(f"Total number of training steps: {num_training_steps*self.world_size}")
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.experiment.model.gamma)
        num_warmup_steps = 0
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )

    
    def setup_loss_fn_dict(self):
        loss_func = build_combined_loss(self.cfg).to(self.local_rank)
        # if self.cfg.host.multi_gpu:
        #     loss_func = DDP(loss_func, device_ids=[self.local_rank])
        self.loss_func = loss_func

    def visualization(self, loader, epoch, coco=None, show=False, num_images=2):
        
        self.model.eval()

        batch = next(iter(loader))
        batch = batch_to_cuda(batch, device=self.cfg.host.device)
        pred = self.model(batch)
        
        outpath = os.path.join(self.cfg.output_dir, "visualizations", f"{epoch}")
        os.makedirs(outpath, exist_ok=True)
        self.logger.info(f"Save visualizations to {outpath}")
        
        coco_anns = defaultdict(list)
        if coco is not None:
            for ann in coco:
                if ann["image_id"] >= num_images: 
                    break
                coco_anns[ann["image_id"]].append(ann)
        
        if self.cfg.experiment.encoder.use_lidar:
            lidar_batches = torch.unbind(batch["lidar"], dim=0)
            
        names = get_tile_names_from_dataloader(loader, batch["image_id"].cpu().numpy().flatten().tolist())

        for i in range(num_images):
        
            fig, ax = plt.subplots(1,2,figsize=(8, 4), dpi=150)
            ax = ax.flatten()

            if self.cfg.experiment.encoder.use_images:

                image = denormalize_image_for_visualization(batch["image"][i], self.cfg)
                plot_image(image, ax=ax[0])
                plot_image(image, ax=ax[1])
            if self.cfg.experiment.encoder.use_lidar:
                plot_point_cloud(lidar_batches[i], ax=ax[0])
                plot_point_cloud(lidar_batches[i], ax=ax[1])
                
            mask_color = [1,0,1,0.6]
            plot_mask(batch["gt_polygons_image"][i][0], ax=ax[0], color=mask_color)
            plot_mask(pred["seg"][i].squeeze()>0.5, ax=ax[1], color=mask_color)
            
            plot_crossfield(batch["gt_crossfield_angle"][i].squeeze(), ax=ax[0])
            pred_crossfield = compute_crossfield_uv(pred["crossfield"][i].permute(1,2,0).detach().cpu().numpy())
            pred_crossfield0 = np.arctan2(pred_crossfield[0].imag, pred_crossfield[0].real)
            plot_crossfield(pred_crossfield0, ax=ax[1])
            ## plot the orthogonal linefield (i.e. the full crossfield)
            # pred_crossfield1 = np.arctan2(pred_crossfield[1].imag, pred_crossfield[1].real)
            # plot_crossfield(pred_crossfield1, ax=ax[1])
            
            polygons = coco_anns[i]
            if len(polygons):
                polygons = coco_anns_to_shapely_polys(polygons)
                plot_shapely_polygons(polygons, ax=ax[1],pointcolor=[1,1,0],edgecolor=[1,0,1])

            ax[0].set_title("GT_"+names[i])
            ax[1].set_title("PRED_"+names[i])
            
            plt.tight_layout()
            outfile = os.path.join(outpath, f"{names[i]}.png")
            self.logger.debug(f"Save visualization to {outfile}")
            plt.savefig(outfile)
            if show:
                plt.show(block=True)
            if self.cfg.run_type.log_to_wandb and self.local_rank == 0:
                wandb.log({f"{epoch}: {names[i]}": wandb.Image(fig)})            
            plt.close(fig)
            
    def valid_one_epoch(self, epoch):

        self.logger.info(f"Validate epoch {epoch}...")
        self.model.eval()
        loss_meter = MetricLogger(" val_")
        loader = self.progress_bar(self.val_loader)

        for batch in loader:
            
            batch = batch_to_cuda(batch, device=self.cfg.host.device)
            pred = self.model(batch)
            loss, loss_dict, extra_dict = self.loss_func(pred, batch, epoch=epoch, normalize=False)

            loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
            loss_reduced = loss.item()
            loss_meter.update(total_loss=loss_reduced, **loss_dict_reduced)
            loader.set_postfix(val_loss=loss_meter.meters["total_loss"].global_avg)

        for k,v in loss_meter.meters.items():
            self.logger.debug(f"Validation {k}: {v.global_avg:.3f}")
            loss_dict[k] = self.average_across_gpus(v)
        self.logger.info(f"Validation loss: {loss_dict['total_loss']:.3f}")

        return loss_dict

    
    def train_one_epoch(self, epoch, iter_idx):
        
        self.logger.info(f"Train epoch {epoch}...")
        self.model.train()
        loss_meter = MetricLogger(" train_")
        loader = self.progress_bar(self.train_loader)
        for batch in loader:
            
            batch = batch_to_cuda(batch, device=self.cfg.host.device)
            pred = self.model(batch)
            loss, loss_dict, extra_dict = self.loss_func(pred, batch, epoch=epoch, normalize=False)
                
            with torch.no_grad():                
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = loss.item()
                loss_meter.update(total_loss=loss_reduced, **loss_dict_reduced)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            lr = get_lr(self.optimizer)
            loader.set_postfix(train_loss=loss_meter.meters["total_loss"].global_avg, lr=f"{lr:.7f}")
            iter_idx += 1
        
        for k,v in loss_meter.meters.items():
            self.logger.debug(f"Train {k}: {v.global_avg:.3f}")
            loss_dict[k] = self.average_across_gpus(v)
        self.logger.info(f"Train loss: {loss_dict['total_loss']:.3f}")

        return loss_dict, iter_idx


    def train_val_loop(self):

        if self.cfg.checkpoint is not None:
            self.load_checkpoint()

        iter_idx=self.cfg.experiment.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.experiment.model.start_epoch, self.cfg.experiment.model.num_epochs)

        predictor = Predictor(self.cfg, local_rank=self.local_rank, world_size=self.world_size)
        if self.local_rank == 0:
            # predictor = Predictor(self.cfg)
            evaluator = Evaluator(self.cfg)
            evaluator.load_gt(self.cfg.dataset.annotations["val"])
        else:
            # predictor = None
            evaluator = None
            
        
        if self.cfg.run_type.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()
            
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
                    self.visualization(self.train_loader,epoch)
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


            ### take the following out of no grad because polygonization of FFL requires grads

            #############################################
            ############## COCO Evaluation ##############
            #############################################
            val_metrics_dict = {}
            if (epoch + 1) % self.cfg.training.val_every == 0:

                self.logger.info("Predict validation set with latest model...")
                coco_predictions = predictor.predict_from_loader(self.model,self.val_loader)
                
                # self.logger.debug(f"rank {self.local_rank}, device: {self.device}, coco_pred_type: {type(coco_predictions)}, coco_pred_len: {len(coco_predictions)}")
                
                if not len(coco_predictions):
                    self.logger.info("No polygons predicted. Skipping coco evaluation...")
                else:
                    ## note that this results in self.local_rank == 0
                    assert isinstance(coco_predictions,dict), f"Coco predictions should be of type dict not {type(coco_predictions)}"
                    poly_method = list(coco_predictions.keys())[0]
                    coco_predictions = list(coco_predictions.values())[0]
                    self.logger.info(f"Evaluate {poly_method} polygonization...")

                    with torch.no_grad():
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


        