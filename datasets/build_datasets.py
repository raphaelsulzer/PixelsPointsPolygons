import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_val_loader(cfg):
    
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
    
    
    val_ds = InriaCocoDataset_val(
        cfg=CFG,
        dataset_dir=args.dataset_dir,
        transform=valid_transforms,
        tokenizer=cfg.tokenizer,
        shuffle_tokens=CFG.SHUFFLE_TOKENS
    )
    
    indices = list(range(1000))
    val_ds = Subset(val_ds, indices)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, max_len=CFG.MAX_LEN, pad_idx=CFG.PAD_IDX),
        num_workers=CFG.NUM_WORKERS
    )
    
    return val_loader