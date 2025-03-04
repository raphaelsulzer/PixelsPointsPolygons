import os
from os import path as osp
import torch
from torch import nn
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import (
    get_linear_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from tokenizer import Tokenizer
from utils_ori import (
    seed_everything,
    load_checkpoint,
)
from ddp_utils import (
    get_inria_loaders,
    get_spacenet_loaders,
    get_whu_buildings_loaders,
    get_mass_roads_loaders,
)

from models.model_ori import (
    Encoder,
    Decoder,
    EncoderDecoder
)

from engine import train_eval

from torch import distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
    dist_url = "env://"  # default

    # only works with torch.distributed.launch or torch.run.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # this will make all .cuda() calls work properly.
    torch.cuda.set_device(local_rank)

    # synchronizes all threads to reach this point before moving on.
    dist.barrier()


def main():
    # setup the process groups
    init_distributed()
    seed_everything(42)

    # Define tensorboard for logging.
    writer = SummaryWriter(f"runs/{CFG.EXPERIMENT_NAME}/logs/tensorboard")
    attrs = vars(CFG)
    with open(f"runs/{CFG.EXPERIMENT_NAME}/config.txt", "w") as f:
        print("\n".join("%s: %s" % item for item in attrs.items()), file=f)

    train_transforms = A.Compose(
        [
            A.Affine(rotate=[-360, 360], fit_output=True, p=0.8),  # scaled rotations are performed before resizing to ensure rotated and scaled images are correctly resized.
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.RandomRotate90(p=1.),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(),
            A.ToGray(p=0.4),
            A.GaussNoise(),
            # ToTensorV2 of albumentations doesn't divide by 255 like in PyTorch,
            # it is done inside Normalize function.
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    valid_transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    if "debug" in CFG.EXPERIMENT_NAME:
        train_transforms = valid_transforms

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    if "inria" in CFG.DATASET:
        train_loader, val_loader, _ = get_inria_loaders(
            CFG.TRAIN_DATASET_DIR,
            CFG.VAL_DATASET_DIR,
            CFG.TEST_IMAGES_DIR,
            tokenizer,
            CFG.MAX_LEN,
            tokenizer.PAD_code,
            CFG.SHUFFLE_TOKENS,
            CFG.BATCH_SIZE,
            train_transforms,
            valid_transforms,
            CFG.NUM_WORKERS,
            CFG.PIN_MEMORY,
        )
    elif "spacenet" in CFG.DATASET:
        train_loader, val_loader, _ = get_spacenet_loaders(
            CFG.TRAIN_DATASET_DIR,
            CFG.VAL_DATASET_DIR,
            CFG.TEST_IMAGES_DIR,
            tokenizer,
            CFG.MAX_LEN,
            tokenizer.PAD_code,
            CFG.SHUFFLE_TOKENS,
            CFG.BATCH_SIZE,
            train_transforms,
            valid_transforms,
            CFG.NUM_WORKERS,
            CFG.PIN_MEMORY,
        )
    elif "whu_buildings" in CFG.DATASET:
        train_loader, val_loader, _ = get_whu_buildings_loaders(
            CFG.TRAIN_DATASET_DIR,
            CFG.VAL_DATASET_DIR,
            CFG.TEST_IMAGES_DIR,
            tokenizer,
            CFG.MAX_LEN,
            tokenizer.PAD_code,
            CFG.SHUFFLE_TOKENS,
            CFG.BATCH_SIZE,
            train_transforms,
            valid_transforms,
            CFG.NUM_WORKERS,
            CFG.PIN_MEMORY,
        )
    elif "mass_roads" in CFG.DATASET:
        train_loader, val_loader, test_loader = get_mass_roads_loaders(
            CFG.TRAIN_DATASET_DIR,
            CFG.VAL_DATASET_DIR,
            CFG.TEST_IMAGES_DIR,
            tokenizer,
            CFG.MAX_LEN,
            tokenizer.PAD_code,
            CFG.SHUFFLE_TOKENS,
            CFG.BATCH_SIZE,
            train_transforms,
            valid_transforms,
            CFG.NUM_WORKERS,
            CFG.PIN_MEMORY,
        )
    else:
        pass

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)

    weight = torch.ones(CFG.PAD_IDX + 1, device=CFG.DEVICE)
    weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=CFG.PAD_IDX, label_smoothing=CFG.LABEL_SMOOTHING, weight=weight)
    perm_loss_fn = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))

    num_training_steps = CFG.NUM_EPOCHS * (len(train_loader.dataset) // CFG.BATCH_SIZE // torch.cuda.device_count())
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    CFG.START_EPOCH = 0
    if CFG.LOAD_MODEL:
        checkpoint_name = osp.basename(osp.realpath(CFG.CHECKPOINT_PATH))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        start_epoch = load_checkpoint(
            torch.load(f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/{checkpoint_name}", map_location=map_location),
            model,
            optimizer,
            lr_scheduler
        )
        CFG.START_EPOCH = start_epoch + 1
        dist.barrier()

    # Convert BatchNorm in model to SyncBatchNorm.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Wrap model with distributed data parallel.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    train_eval(
        model,
        train_loader,
        val_loader,
        val_loader,
        tokenizer,
        vertex_loss_fn,
        perm_loss_fn,
        optimizer,
        lr_scheduler=lr_scheduler,
        step='batch',
        writer=writer
    )


if __name__ == "__main__":
    main()

