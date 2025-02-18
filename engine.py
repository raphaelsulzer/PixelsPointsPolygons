from tqdm import tqdm
import torch
from copy import deepcopy

from utils import (
    AverageMeter,
    get_lr,
    save_checkpoint,
    save_single_predictions_as_images
)
from config import CFG

from ddp_utils import is_main_process


def train_one_epoch(epoch, iter_idx, model, train_loader, optimizer, lr_scheduler, vertex_loss_fn, perm_loss_fn, writer):
    model.train()
    vertex_loss_fn.train()
    perm_loss_fn.train()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = train_loader
    if is_main_process():
        loader = tqdm(train_loader, total=len(train_loader))

    for x, y_mask, y_corner_mask, y, y_perm in loader:
        x = x.to(CFG.DEVICE, non_blocking=True)
        y = y.to(CFG.DEVICE, non_blocking=True)
        y_perm = y_perm.to(CFG.DEVICE, non_blocking=True)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        preds, perm_mat = model(x, y_input)

        if epoch < CFG.MILESTONE:
            vertex_loss_weight = CFG.vertex_loss_weight
            perm_loss_weight = 0.0
        else:
            vertex_loss_weight = CFG.vertex_loss_weight
            perm_loss_weight = CFG.perm_loss_weight

        vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
        perm_loss = perm_loss_weight*perm_loss_fn(perm_mat, y_perm)

        loss = vertex_loss + perm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=0.1)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        vertex_loss_meter.update(vertex_loss.item(), x.size(0))
        perm_loss_meter.update(perm_loss.item(), x.size(0))

        lr = get_lr(optimizer)
        if is_main_process():
            loader.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
            # print(f"Running_logs/Train_Loss: {loss_meter.avg}")
            # writer.add_scalar('Running_logs/Train_Loss', loss_meter.avg, iter_idx)
            # writer.add_scalar('Running_logs/LR', lr, iter_idx)
            # writer.add_image(f"Running_logs/input_images", torchvision.utils.make_grid(x), iter_idx)
            # writer.add_graph(model, input_to_model=(x, y_input))

        iter_idx += 1

        # if iter_idx % 5 == 0:
        #     break
        # prof.step()
    # prof.stop()
    print(f"Total train loss: {loss_meter.avg}\n\n")
    loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict, iter_idx


def valid_one_epoch(epoch, model, valid_loader, vertex_loss_fn, perm_loss_fn):
    print(f"\nValidating...")
    model.eval()
    vertex_loss_fn.eval()
    perm_loss_fn.eval()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = valid_loader
    if is_main_process():
        loader = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for x, y_mask, y_corner_mask, y, y_perm in loader:
            x = x.to(CFG.DEVICE, non_blocking=True)
            y = y.to(CFG.DEVICE, non_blocking=True)
            y_perm = y_perm.to(CFG.DEVICE, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds, perm_mat = model(x, y_input)

            if epoch < CFG.MILESTONE:
                vertex_loss_weight = CFG.vertex_loss_weight
                perm_loss_weight = 0.0
            else:
                vertex_loss_weight = CFG.vertex_loss_weight
                perm_loss_weight = CFG.perm_loss_weight
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


def train_eval(
    model,
    train_loader,
    valid_loader,
    test_loader,
    tokenizer,
    vertex_loss_fn,
    perm_loss_fn,
    optimizer,
    lr_scheduler,
    step,
    writer
):
    best_loss = float('inf')
    best_metric = float('-inf')

    iter_idx=CFG.START_EPOCH * len(train_loader)
    epoch_iterator = range(CFG.START_EPOCH, CFG.NUM_EPOCHS)
    if is_main_process():
        epoch_iterator = tqdm(epoch_iterator)
    for epoch in epoch_iterator:
        if is_main_process():
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
            writer
        )

        wandb_dict = deepcopy(train_loss_dict)
        wandb_dict['epoch'] = epoch

        if is_main_process():
            writer.add_scalar('Train_Losses/Total_Loss', train_loss_dict['total_loss'], epoch)
            writer.add_scalar('Train_Losses/Vertex_Loss', train_loss_dict['vertex_loss'], epoch)
            writer.add_scalar('Train_Losses/Perm_Loss', train_loss_dict['perm_loss'], epoch)

        valid_loss_dict = valid_one_epoch(
            epoch,
            model,
            valid_loader,
            vertex_loss_fn,
            perm_loss_fn,
        )  # TODO: add eval metrics to validation function?
        if is_main_process():
            print(f"Valid loss: {valid_loss_dict['total_loss']:.3f}\n\n")

        # Save best validation loss epoch.
        if valid_loss_dict['total_loss'] < best_loss and CFG.SAVE_BEST and is_main_process():
            best_loss = valid_loss_dict['total_loss']
            checkpoint = {
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                filename="best_valid_loss.pth"
            )
            # torch.save(model.state_dict(), 'best_valid_loss.pth')
            print(f"Saved best val loss model.")

        # Save latest checkpoint every epoch.
        if CFG.SAVE_LATEST and is_main_process():
            checkpoint = {
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
            save_checkpoint(
                checkpoint,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                filename="latest.pth"
            )

        if (epoch + 1) % CFG.SAVE_EVERY == 0 and is_main_process():
            checkpoint = {
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                filename=f"epoch_{epoch}.pth"
            )

        if is_main_process():
            writer.add_scalar('Val_Losses/Total_Loss', valid_loss_dict['total_loss'], epoch)
            writer.add_scalar('Val_Losses/Vertex_Loss', valid_loss_dict['vertex_loss'], epoch)
            writer.add_scalar('Val_Losses/Perm_Loss', valid_loss_dict['perm_loss'], epoch)

        # output examples to a folder
        if (epoch + 1) % CFG.VAL_EVERY == 0 and is_main_process():
            val_metrics_dict = save_single_predictions_as_images(
                test_loader,
                model,
                tokenizer,
                epoch,
                writer,
                folder=f"runs/{CFG.EXPERIMENT_NAME}/runtime_outputs/",
                device=CFG.DEVICE
            )
            for metric, value in zip(val_metrics_dict.keys(), val_metrics_dict.values()):
                print(f"{metric}: {value}")

            # Save best single batch validation metric epoch.
            if val_metrics_dict["miou"] > best_metric and CFG.SAVE_BEST and is_main_process():
                best_metric = val_metrics_dict["miou"]
                checkpoint = {
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
                save_checkpoint(
                    checkpoint,
                    folder=f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/",
                    filename="best_valid_metric.pth"
                )
                print(f"Saved best val metric model.")

