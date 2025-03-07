from tqdm import tqdm
import torch, os
from postprocess_coco_parts import *

from utils import AverageMeter, get_lr, save_checkpoint, save_single_predictions_as_images

from lidar_poly_dataset.metrics import compute_IoU_cIoU

import wandb

def setup_wandb(cfg):

    log_outfile = os.path.join(cfg.output_dir, 'log.txt')

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="HiSup",
        name=cfg.experiment_name,
        group="v1_pix2poly",
        # track hyperparameters and run metadata
        config=cfg
    )

    wandb.run.log_code(log_outfile)


def valid_one_epoch(epoch, model, valid_loader, vertex_loss_fn, perm_loss_fn, cfg):
    print(f"\nValidating...")
    model.eval()
    vertex_loss_fn.eval()
    perm_loss_fn.eval()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = valid_loader
    loader = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for x, y_mask, y_corner_mask, y, y_perm in loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            y_perm = y_perm.to(cfg.device, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds, perm_mat = model(x, y_input)

            if epoch < cfg.model.milestone:
                vertex_loss_weight = cfg.model.vertex_loss_weight
                perm_loss_weight = 0.0
            else:
                vertex_loss_weight = cfg.model.vertex_loss_weight
                perm_loss_weight = cfg.model.perm_loss_weight
            vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            perm_loss = perm_loss_weight*perm_loss_fn(perm_mat, y_perm)

            loss = vertex_loss + perm_loss

            loss_meter.update(loss.item(), x.size(0))
            vertex_loss_meter.update(vertex_loss.item(), x.size(0))
            perm_loss_meter.update(perm_loss.item(), x.size(0))

        loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict


def train_one_epoch(epoch, iter_idx, model, train_loader, optimizer, lr_scheduler, vertex_loss_fn, perm_loss_fn, cfg):
    model.train()
    vertex_loss_fn.train()
    perm_loss_fn.train()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = train_loader
    loader = tqdm(train_loader, total=len(train_loader))

    for x, y_mask, y_corner_mask, y, y_perm, id in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        y_perm = y_perm.to(cfg.device, non_blocking=True)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        preds, perm_mat = model(x, y_input)

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

        loss_meter.update(loss.item(), x.size(0))
        vertex_loss_meter.update(vertex_loss.item(), x.size(0))
        perm_loss_meter.update(perm_loss.item(), x.size(0))

        lr = get_lr(optimizer)

        loader.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
        # print(f"Running_logs/Train_Loss: {loss_meter.avg}")
        # writer.add_scalar('Running_logs/Train_Loss', loss_meter.avg, iter_idx)
        # writer.add_scalar('Running_logs/LR', lr, iter_idx)
        # writer.add_image(f"Running_logs/input_images", torchvision.utils.make_grid(x), iter_idx)
        # writer.add_graph(model, input_to_model=(x, y_input))

        iter_idx += 1

        # if iter_idx % 50 == 0:
        #     break

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
        setup_wandb()

    best_loss = float('inf')
    best_metric = float('-inf')

    iter_idx=cfg.model.start_epoch * len(train_loader)
    epoch_iterator = range(cfg.model.start_epoch, cfg.model.num_epochs)


    for epoch in tqdm(epoch_iterator, position=0, leave=True):
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

        wandb_dict ={}
        wandb_dict['epoch'] = epoch
        for k, v in train_loss_dict.items():
            wandb_dict[f"train_{k}"] = v

        val_loss_dict = valid_one_epoch(
            epoch,
            model,
            val_loader,
            vertex_loss_fn,
            perm_loss_fn,
            cfg=cfg
        )

        for k, v in val_loss_dict.items():
            wandb_dict[f"val_{k}"] = v

        # Save best validation loss epoch.
        if val_loss_dict['total_loss'] < best_loss and cfg.model.save_best:
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
                folder=os.path.join(cfg.output_dir,"logs","checkpoints"),
                filename="validation_best.pth"
            )
            print(f"Saved best val loss model.")

        # Save latest checkpoint every epoch.
        if cfg.model.save_latest:
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

        if (epoch + 1) % cfg.model.save_every == 0:
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
        if (epoch + 1) % cfg.model.val_every == 0:

            save_single_predictions_as_images(
                val_loader,
                model,
                tokenizer,
                epoch,
                wandb_dict,
                folder=os.path.join(cfg.output_dir, "runtime_outputs")
            )

            # batched_polygons = predict_to_coco(model, tokenizer, val_loader)
            # outfile = os.path.join(cfg.output_dir,"coco",f"validation_{epoch}.json")
            # combine_polygons_from_list(batched_polygons, outfile)
            #
            # iou, ciou = compute_IoU_cIoU(outfile, val_loader.dataset.ann_file)
            # print("Iou: {:.4f}, CIou: {:.4f}".format())

            # # Save best single batch validation metric epoch.
            # if wandb_dict["miou"] > best_metric and CFG.SAVE_BEST:
            #     best_metric = wandb_dict["miou"]
            #     checkpoint = {
            #         "state_dict": model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "scheduler": lr_scheduler.state_dict(),
            #         "epochs_run": epoch,
            #         "loss": train_loss_dict["total_loss"]
            #     }
            #     save_checkpoint(
            #         checkpoint,
            #         folder=os.path.join(cfg.output_dir, "logs", "checkpoints"),
            #         filename="validation_best.pth"
            #     )
            #     print(f"Saved best val metric model.")

        for k,v in wandb_dict.items():
            print(f"{k}: {v}")

        if cfg.log_to_wandb:
            wandb.log(wandb_dict)


