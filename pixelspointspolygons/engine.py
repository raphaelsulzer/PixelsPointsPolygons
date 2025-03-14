import sys
import os
import torch
import wandb
import json

from tqdm import tqdm

from .misc import *
from .misc.ddp_utils import is_main_process

from .eval import evaluate
from .predictor import Predictor

def valid_one_epoch(epoch, model, valid_loader, vertex_loss_fn, perm_loss_fn, cfg):
    print(f"\nValidating...")
    model.eval()
    vertex_loss_fn.eval()
    perm_loss_fn.eval()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = valid_loader
    loader = tqdm(valid_loader, total=len(valid_loader), file=sys.stdout, dynamic_ncols=True, mininterval=20.0)

    with torch.no_grad():
        for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in loader:
            x_image = x_image.to(cfg.device, non_blocking=True)
            y_sequence = y_sequence.to(cfg.device, non_blocking=True)
            y_perm = y_perm.to(cfg.device, non_blocking=True)

            y_input = y_sequence[:, :-1]
            y_expected = y_sequence[:, 1:]

            preds, perm_mat = model(x_image, x_lidar, y_input)

            if epoch < cfg.model.milestone:
                vertex_loss_weight = cfg.model.vertex_loss_weight
                perm_loss_weight = 0.0
            else:
                vertex_loss_weight = cfg.model.vertex_loss_weight
                perm_loss_weight = cfg.model.perm_loss_weight
            vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            perm_loss = perm_loss_weight*perm_loss_fn(perm_mat, y_perm)

            loss = vertex_loss + perm_loss

            loss_meter.update(loss.item(), x_image.size(0))
            vertex_loss_meter.update(vertex_loss.item(), x_image.size(0))
            perm_loss_meter.update(perm_loss.item(), x_image.size(0))

        loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict


def train_one_epoch(epoch, iter_idx, 
                    model, 
                    train_loader, optimizer, lr_scheduler, vertex_loss_fn, perm_loss_fn, cfg):
    model.train()
    vertex_loss_fn.train()
    perm_loss_fn.train()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = train_loader
    loader = tqdm(train_loader, total=len(train_loader), file=sys.stdout, dynamic_ncols=True, mininterval=20.0)

    for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in loader:
        
        # ### debug vis
        if cfg.debug_vis:
            file_names = get_image_file_name_from_dataloader(train_loader.dataset.coco.imgs, image_ids)
            plot_pix2poly(image_batch=x_image,mask_batch=y_mask,corner_image_batch=y_corner_mask,file_names=file_names)        
        
        if cfg.use_images:
            x_image = x_image.to(cfg.device, non_blocking=True)
        if cfg.use_lidar:
            x_lidar = x_lidar.to(cfg.device, non_blocking=True)
        
        y_sequence = y_sequence.to(cfg.device, non_blocking=True)
        y_perm = y_perm.to(cfg.device, non_blocking=True)

        y_input = y_sequence[:, :-1]
        y_expected = y_sequence[:, 1:]

        preds, perm_mat = model(x_image, x_lidar, y_input)

        if epoch < cfg.model.milestone:
            vertex_loss_weight = cfg.model.vertex_loss_weight
            perm_loss_weight = 0.0
        else:
            vertex_loss_weight = cfg.model.vertex_loss_weight
            perm_loss_weight = cfg.model.perm_loss_weight

        vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
        perm_loss = perm_loss_weight*perm_loss_fn(perm_mat, y_perm)

        loss = vertex_loss + perm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), x_image.size(0))
        vertex_loss_meter.update(vertex_loss.item(), x_image.size(0))
        perm_loss_meter.update(perm_loss.item(), x_image.size(0))

        lr = get_lr(optimizer)

        loader.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")

        iter_idx += 1

        if cfg.run_type.name=="debug" and iter_idx % 10 == 0:
            break

    print(f"Total train loss: {loss_meter.avg}\n\n")
    loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict, iter_idx



def train_eval(
    model,
    train_loader,
    val_loader,
    tokenizer,
    vertex_loss_fn,
    perm_loss_fn,
    optimizer,
    lr_scheduler,
    step,
    cfg
):

    if cfg.log_to_wandb:
        if is_main_process():
            init_wandb(cfg)

    best_loss = float('inf')

    iter_idx=cfg.model.start_epoch * len(train_loader)
    epoch_iterator = range(cfg.model.start_epoch, cfg.model.num_epochs)

    pp = Predictor(cfg)

    for epoch in tqdm(epoch_iterator, position=0, leave=True, file=sys.stdout, dynamic_ncols=True, mininterval=20.0):
        print(f"\n\nEPOCH: {epoch + 1}\n\n")

        train_loss_dict, iter_idx = train_one_epoch(
            epoch,
            iter_idx,
            model,
            train_loader,
            optimizer,
            lr_scheduler if step=='batch' else None,
            vertex_loss_fn,
            perm_loss_fn,
            cfg=cfg
        )
        if is_main_process():
            wandb_dict ={}
            wandb_dict['epoch'] = epoch
            for k, v in train_loss_dict.items():
                wandb_dict[f"train_{k}"] = v
            wandb_dict['lr'] = get_lr(optimizer)

        val_loss_dict = valid_one_epoch(
            epoch,
            model,
            val_loader,
            vertex_loss_fn,
            perm_loss_fn,
            cfg=cfg
        )
        if is_main_process():
            for k, v in val_loss_dict.items():
                wandb_dict[f"val_{k}"] = v
            print(f"Valid loss: {val_loss_dict['total_loss']:.3f}\n\n")

        # Save best validation loss epoch.
        if val_loss_dict['total_loss'] < best_loss and cfg.save_best:
            best_loss = val_loss_dict['total_loss']
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=os.path.join(cfg.output_dir,"checkpoints"),
                filename="validation_best.pth"
            )
            print(f"Saved best val loss model.")

        # Save latest checkpoint every epoch.
        if cfg.save_latest:
            checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
            save_checkpoint(
                checkpoint,
                folder=os.path.join(cfg.output_dir,"logs","checkpoints"),
                filename="latest.pth"
            )

        if (epoch + 1) % cfg.save_every == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=os.path.join(cfg.output_dir,"logs","checkpoints"),
                filename=f"epoch_{epoch}.pth"
            )

        # output examples to a folder
        if (epoch + 1) % cfg.val_every == 0:
            
            # val_metrics_dict = save_single_predictions_as_images(val_loader, model, tokenizer,
            #     epoch,
            #     folder=os.path.join(cfg.output_dir, "runtime_outputs"),
            #     cfg=cfg
            # )
            
            coco_predictions = pp.predict_from_loader(model,tokenizer,val_loader)

            if len(coco_predictions) > 0:
                
                try:
                    print("Found some predictions. Evaluating...")
                    
                    prediction_outfile = os.path.join(cfg.output_dir, "predictions", f"epoch_{epoch}.json")
                    os.makedirs(os.path.dirname(prediction_outfile), exist_ok=True)
                    with open(prediction_outfile, "w") as fp:
                        fp.write(json.dumps(coco_predictions))
                    
                    val_metrics_dict = evaluate(val_loader.dataset.ann_file, prediction_outfile, modes=cfg.eval.modes)

                    for metric, value in val_metrics_dict.items():
                        if is_main_process():
                            wandb_dict[f"val_{metric}"] = value
                            
                except Exception as e:
                    print(f"Error evaluating predictions: {e}")
                    


        if cfg.log_to_wandb:
            if is_main_process():
                wandb.log(wandb_dict)


