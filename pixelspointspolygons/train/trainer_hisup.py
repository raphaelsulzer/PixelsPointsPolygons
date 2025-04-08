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

from torch import optim
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict

from ..misc import get_lr, get_tile_names_from_dataloader, MetricLogger
from ..models.hisup import HiSupModel
from ..eval import Evaluator
from ..misc.coco_conversions import generate_coco_ann
from ..misc.debug_visualisations import *
from ..misc.coco_conversions import coco_anns_to_shapely_polys


from .trainer import Trainer


class LossReducer:
    
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.model.loss_weights)

    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k] * loss_dict[k]
                          for k in self.loss_weights.keys()])

        return total_loss

class HiSupTrainer(Trainer):
    
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
        self.model = HiSupModel(self.cfg, self.local_rank)


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
        
        self.loss_reducer = LossReducer(self.cfg)
    
    
    def visualization(self, loader, epoch, coco=None, num_images=2):
        
        self.model.eval()
        
        x_image, x_lidar, y, tile_ids = next(iter(loader))
        if self.cfg.use_images:
            x_image = x_image.to(self.cfg.device, non_blocking=True)
        if self.cfg.use_lidar:
            x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)
            
        y=self.to_single_device(y,self.cfg.device)

        polygon_output, loss_dict = self.model(x_image, x_lidar, y)
        polygon_output = self.to_single_device(polygon_output, 'cpu')
        
        
        outpath = os.path.join(self.cfg.output_dir, "visualizations", f"{epoch}")
        os.makedirs(outpath, exist_ok=True)
        self.logger.info(f"Save visualizations to {outpath}")
        
        coco_anns = defaultdict(list)
        if coco is not None:
            for ann in coco:
                if ann["image_id"] >= num_images: 
                    break
                coco_anns[ann["image_id"]].append(ann)
                
        if self.cfg.use_lidar:
            lidar_batches = torch.unbind(x_lidar, dim=0)
            
        names = get_tile_names_from_dataloader(loader, tile_ids.cpu().numpy().flatten().tolist())

        for i in range(num_images):
        
            fig, ax = plt.subplots(1,2,figsize=(8, 4), dpi=150)
            ax = ax.flatten()

            if self.cfg.use_images:
                image = (x_image[i].permute(1, 2, 0).cpu().numpy()*np.array(self.cfg.dataset.image_std) + np.array(self.cfg.dataset.image_mean))
                image = np.clip(image/255.0, 0, 1)
                plot_image(image, ax=ax[0])
                plot_image(image, ax=ax[1])
            if self.cfg.use_lidar:
                plot_point_cloud(lidar_batches[i], ax=ax[0])
                plot_point_cloud(lidar_batches[i], ax=ax[1])
                
            mask_color = [1,0,1,0.6]
            plot_mask(y[i]["mask_ori"], ax=ax[0], color=mask_color)
            plot_mask(polygon_output["mask_pred"][i]>0.5, ax=ax[1], color=mask_color)
            
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
            if self.cfg.log_to_wandb and self.local_rank == 0:
                wandb.log({f"{epoch}: {names[i]}": wandb.Image(fig)})            
            plt.close(fig)
    
    
    def valid_one_epoch(self):

        self.logger.info("Validate...")
        self.model.eval()

        loss_meter = MetricLogger(" val_")

        loader = self.progress_bar(self.val_loader)

        coco_predictions = []
        
        for x_image, x_lidar, y, tile_ids in loader:
            
            batch_size = x_image.size(0) if self.cfg.use_images else x_lidar.size(0)
            
            if self.cfg.use_images:
                x_image = x_image.to(self.cfg.device, non_blocking=True)
            if self.cfg.use_lidar:
                x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)
                
            y=self.to_single_device(y,self.cfg.device)

            polygon_output, loss_dict = self.model(x_image, x_lidar, y)
            
            ## loss stuff
            loss = self.loss_reducer(loss_dict)
            loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
            loss_reduced = loss.item()
            loss_meter.update(total_loss=loss_reduced, **loss_dict_reduced)
            loader.set_postfix(val_loss=loss_meter.meters["total_loss"].global_avg)
            ## polygon stuff
            polygon_output = self.to_single_device(polygon_output, 'cpu')
            batch_scores = polygon_output['scores']
            batch_polygons = polygon_output['polys_pred']

            for b in range(batch_size):

                scores = batch_scores[b]
                polys = batch_polygons[b]

                image_result = generate_coco_ann(polys, tile_ids[b], scores=scores)
                if len(image_result) != 0:
                    coco_predictions.extend(image_result)

        self.logger.debug(f"Validation loss: {loss_meter.meters['total_loss'].global_avg:.3f}")
        
        for k,v in loss_meter.meters.items():
            loss_dict[k] = self.average_across_gpus(v)
        
        self.logger.info(f"Validation loss: {loss_dict['total_loss']:.3f}")

        return loss_dict, coco_predictions



    def train_one_epoch(self, epoch, iter_idx):
        
        self.logger.info(f"Train epoch {epoch}...")
        
        self.model.train()

        loss_meter = MetricLogger(" train_")

        loader = self.progress_bar(self.train_loader)

        for x_image, x_lidar, y, tile_ids in loader:
                        

            y=self.to_single_device(y,self.cfg.device)
            
            if self.cfg.use_images:
                x_image = x_image.to(self.cfg.device, non_blocking=True)
            if self.cfg.use_lidar:
                x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)
            
            loss_dict = self.model(x_image, x_lidar, y)
            loss = self.loss_reducer(loss_dict)

            with torch.no_grad():
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

            # if self.cfg.run_type.name=="debug" and iter_idx % 10 == 0:
            # break
            
        
        self.logger.debug(f"Train loss: {loss_meter.meters['total_loss'].global_avg:.3f}")
        
        for k,v in loss_meter.meters.items():
            loss_dict[k] = self.average_across_gpus(v)
        
        self.logger.info(f"Train loss: {loss_dict['total_loss']:.3f}")

        return loss_dict, iter_idx


    def train_val_loop(self):

        if self.cfg.checkpoint is not None or self.cfg.checkpoint_file is not None:
            self.load_checkpoint()
            
        if self.cfg.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()

        best_loss = float('inf')

        iter_idx=self.cfg.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.model.start_epoch, self.cfg.model.num_epochs)

        if self.local_rank == 0:
            evaluator = Evaluator(self.cfg)
            evaluator.load_gt()
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
                val_loss_dict, coco_predictions = self.valid_one_epoch()
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

                    self.logger.info("Evaluate validation set with latest model...")

                    
                    if self.is_ddp:
                        
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
                        evaluator.print_dict_results(val_metrics_dict)

                        for metric, value in val_metrics_dict.items():
                            wandb_dict[f"val_{metric}"] = value
                        

                self.logger.info("Validation finished...\n")
                    
                # Sync all processes before next epoch
                if self.is_ddp:
                    dist.barrier()

                if self.cfg.log_to_wandb:
                    if self.local_rank == 0:
                        wandb.log(wandb_dict)

    

