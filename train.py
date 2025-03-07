import os
import torch
from torch import nn
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import get_linear_schedule_with_warmup
import wandb
import hydra
from omegaconf import OmegaConf

from config import CFG
from tokenizer import Tokenizer
from utils import seed_everything, load_checkpoint
from ddp_utils import get_lidar_poly_loaders, get_inria_loaders

from models.model import Encoder, Decoder, EncoderDecoder
from engine import train_eval



def get_model(cfg,tokenizer):
    
    encoder = Encoder(model_name=cfg.model.type, pretrained=True, out_dim=256)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        encoder_len=cfg.model.num_patches,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_len=cfg.model.tokenizer.max_len,
        pad_idx=cfg.model.tokenizer.pad_idx,
    )
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        n_vertices=cfg.model.tokenizer.n_vertices,
        sinkhorn_iterations=cfg.model.sinkhorn_iterations
    )
    model.to(cfg.device)
    model.eval()
    model_taking_encoded_images = EncoderDecoderWithAlreadyEncodedImages(model)
    model_taking_encoded_images.to(cfg.device)
    model_taking_encoded_images.eval()
    
    return model, model_taking_encoded_images



def run_training(cfg):
    # Set random seeds for reproducibility
    seed_everything(42)

    train_transforms = A.ReplayCompose([
        A.D4(p=1.0),
        # A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
        # A.RandomRotate90(p=1.),
        # A.RandomBrightnessContrast(p=0.5), # ColorJitter already does that
        A.ColorJitter(p=0.5),
        # A.ToGray(p=0.4),
        # A.GaussNoise(),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format='yx')
    )

    val_transforms = A.Compose([
        # A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format='yx')
    )

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    if "lidar_poly" in CFG.DATASET:
        train_loader, val_loader, _ = get_lidar_poly_loaders(
            CFG.TRAIN_DATASET_DIR,
            CFG.VAL_DATASET_DIR,
            CFG.TEST_IMAGES_DIR,
            tokenizer,
            CFG.N_VERTICES,
            CFG.MAX_LEN,
            tokenizer.PAD_code,
            CFG.SHUFFLE_TOKENS,
            CFG.BATCH_SIZE,
            train_transforms,
            val_transforms,
            CFG.NUM_WORKERS,
            CFG.PIN_MEMORY
        )
    elif "inria" in CFG.DATASET:
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
            val_transforms,
            CFG.NUM_WORKERS,
            CFG.PIN_MEMORY
        )
    else:
        raise NotImplementedError

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(cfg=CFG, vocab_size=tokenizer.vocab_size, encoder_len=CFG.NUM_PATCHES, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params/10**6:.2f}M parameters")

    weight = torch.ones(CFG.PAD_IDX + 1, device=CFG.DEVICE)
    weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=CFG.PAD_IDX, label_smoothing=CFG.LABEL_SMOOTHING, weight=weight)
    perm_loss_fn = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))

    num_training_steps = CFG.NUM_EPOCHS * (len(train_loader.dataset) // CFG.BATCH_SIZE)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    if CFG.LOAD_MODEL:
        checkpoint_name = os.path.basename(os.path.realpath(CFG.CHECKPOINT_PATH))
        start_epoch = load_checkpoint(
            torch.load(f"runs/{CFG.EXPERIMENT_NAME}/logs/checkpoints/{checkpoint_name}"),
            model,
            optimizer,
            lr_scheduler
        )
        CFG.START_EPOCH = start_epoch + 1

    train_eval(
        model,
        train_loader,
        val_loader,
        tokenizer,
        vertex_loss_fn,
        perm_loss_fn,
        optimizer,
        lr_scheduler=lr_scheduler,
        step='batch'
    )

    # wandb.finish()

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    run_training(cfg)

if __name__ == "__main__":
    main()