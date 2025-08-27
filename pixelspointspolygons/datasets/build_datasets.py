import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from .p3_coco import TestDataset, ValDataset, TrainDataset
from .collate_funcs import collate_fn_pix2poly, collate_fn_hisup, collate_fn_ffl

def get_collate_fn(model):
    
    if model == "pix2poly":
        return collate_fn_pix2poly
    elif model == "hisup":
        return collate_fn_hisup
    elif model == "ffl":
        return collate_fn_ffl
    else:
        raise NotImplementedError(f"Collate function for {model} not implemented yet.")


def get_test_loader(cfg,tokenizer=None,logger=None):
    if cfg.experiment.dataset.name == 'inria':
        raise NotImplementedError
    elif cfg.experiment.dataset.name in ['lidarpoly','p3',"PixelsPointsPolygons","RoofGraphDataset"]:
        return get_test_loader_lidarpoly(cfg,tokenizer,logger)
    else:
        raise NotImplementedError
    
    
def get_val_loader(cfg,tokenizer=None,logger=None):
    if cfg.experiment.dataset.name == 'inria':
        return get_val_loader_inria(cfg,tokenizer,logger)
    elif cfg.experiment.dataset.name in ['lidarpoly','p3',"PixelsPointsPolygons","RoofGraphDataset"]:
        return get_val_loader_lidarpoly(cfg,tokenizer,logger)
    else:
        raise NotImplementedError

def get_train_loader(cfg,tokenizer=None,logger=None):
    if cfg.experiment.dataset.name == 'inria':
        return get_train_loader_inria(cfg,tokenizer,logger)
    elif cfg.experiment.dataset.name in ['lidarpoly','p3',"PixelsPointsPolygons","RoofGraphDataset"]:
        return get_train_loader_p3(cfg,tokenizer,logger)
    else:
        raise NotImplementedError

def get_train_loader_p3(cfg,tokenizer=None,logger=None):
    
    transforms = []
    if cfg.experiment.encoder.augmentations is not None:
        if "D4" in cfg.experiment.encoder.augmentations:
            transforms.append(A.D4(p=1.0))
        if "Resize" in cfg.experiment.encoder.augmentations:
            transforms.append(A.Resize(height=cfg.experiment.encoder.in_height, width=cfg.experiment.encoder.in_width))
        if "ColorJitter" in cfg.experiment.encoder.augmentations:
            transforms.append(A.ColorJitter())
        if "GaussNoise" in cfg.experiment.encoder.augmentations:
            transforms.append(A.GaussNoise())
        if "Normalize" in cfg.experiment.encoder.augmentations:
            # TODO:
            # check what to do for ImageNet normalization for UNetResNet: https://pytorch.org/vision/stable/models.html
            # and also this has to probably be removed for ViT. or check what is the correct way for that.
            transforms.append(A.Normalize(mean=cfg.experiment.encoder.image_mean, std=cfg.experiment.encoder.image_std, max_pixel_value=cfg.experiment.encoder.image_max_pixel_value)) 
    
    transforms.append(ToTensorV2())
    train_transforms = A.ReplayCompose(transforms=transforms,
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )
    if logger is not None:
        for t in transforms:
            logger.debug(f"Added transform {t} to training pipeline.")
    
    train_ds = TrainDataset(
        cfg,
        transform=train_transforms,
        tokenizer=tokenizer
    )
    if cfg.experiment.dataset.train_subset is not None:
        indices = list(range(cfg.experiment.dataset.train_subset))
        ann_file = train_ds.ann_file
        coco = train_ds.coco
        split = train_ds.split
        train_ds = Subset(train_ds, indices)
        train_ds.ann_file = ann_file
        train_ds.coco = coco    
        train_ds.split = split
    
    if logger is not None:
        logger.debug(f"Train dataset created with {len(train_ds)} image/lidar samples.")
        
    sampler = DistributedSampler(dataset=train_ds, shuffle=cfg.run_type.name!='debug') if cfg.host.multi_gpu else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.experiment.model.batch_size,
        collate_fn=partial(get_collate_fn(cfg.experiment.model.name), cfg=cfg),
        num_workers=cfg.num_workers,
        pin_memory=cfg.run_type.name!='debug',
        drop_last=False,
        sampler=sampler,
        shuffle=(sampler is None) and (cfg.run_type.name != 'debug')
    )
    if logger is not None:
        logger.debug(f"Train loader created with {len(train_loader)} batches of size {cfg.experiment.model.batch_size}.")
    
    return train_loader

def get_val_loader_lidarpoly(cfg,tokenizer=None,logger=None):
    
    transforms = []
    if cfg.experiment.encoder.augmentations is not None:
        if "Resize" in cfg.experiment.encoder.augmentations:
            transforms.append(A.Resize(height=cfg.experiment.encoder.in_height, width=cfg.experiment.encoder.in_width))
        if "Normalize" in cfg.experiment.encoder.augmentations:
            transforms.append(A.Normalize(mean=cfg.experiment.encoder.image_mean, std=cfg.experiment.encoder.image_std, max_pixel_value=cfg.experiment.encoder.image_max_pixel_value)) 
            
    transforms.append(ToTensorV2())
    val_transforms = A.ReplayCompose(transforms=transforms,
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )
    
    val_ds = ValDataset(
        cfg,
        transform=val_transforms,
        tokenizer=tokenizer
    )
    
    if logger is not None:
        logger.debug(f"Val dataset created with {len(val_ds)} samples.")
    
    if cfg.experiment.dataset.val_subset is not None:
        indices = list(range(cfg.experiment.dataset.val_subset))
        ann_file = val_ds.ann_file
        coco = val_ds.coco
        split = val_ds.split
        val_ds = Subset(val_ds, indices)
        val_ds.ann_file = ann_file
        val_ds.coco = coco
        val_ds.split = split

    sampler = DistributedSampler(dataset=val_ds, shuffle=False) if cfg.host.multi_gpu else None
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.experiment.model.batch_size,
        collate_fn=partial(get_collate_fn(cfg.experiment.model.name), cfg=cfg),
        num_workers=cfg.num_workers,
        pin_memory=cfg.run_type.name!='debug',
        drop_last=False,
        sampler=sampler,
        shuffle=False
    )
    if logger is not None:
        logger.debug(f"Val loader created with {len(val_loader)} batches of size {cfg.experiment.model.batch_size}.")
    
    return val_loader



def get_test_loader_lidarpoly(cfg,tokenizer=None,logger=None):
    
    transforms = []
    if "Resize" in cfg.experiment.encoder.augmentations:
        transforms.append(A.Resize(height=cfg.experiment.encoder.in_height, width=cfg.experiment.encoder.in_width))
    if "Normalize" in cfg.experiment.encoder.augmentations:
        transforms.append(A.Normalize(mean=cfg.experiment.encoder.image_mean, std=cfg.experiment.encoder.image_std, max_pixel_value=cfg.experiment.encoder.image_max_pixel_value)) 
    transforms.append(ToTensorV2())
    transforms = A.ReplayCompose(transforms=transforms,
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )
    
    ds = TestDataset(
        cfg,
        transform=transforms,
        tokenizer=tokenizer
    )
    
    if logger is not None:
        logger.debug(f"Test dataset created with {len(ds)} samples.")
    
    if cfg.experiment.dataset.test_subset is not None:
        indices = list(range(cfg.experiment.dataset.test_subset))
        ann_file = ds.ann_file
        coco = ds.coco
        split = ds.split
        ds = Subset(ds, indices)
        ds.ann_file = ann_file
        ds.coco = coco
        ds.split = split

    sampler = DistributedSampler(dataset=ds, shuffle=False) if cfg.host.multi_gpu else None
    
    loader = DataLoader(
        ds,
        batch_size=cfg.experiment.model.batch_size,
        collate_fn=partial(get_collate_fn(cfg.experiment.model.name), cfg=cfg),
        num_workers=cfg.num_workers,
        pin_memory=cfg.run_type.name!='debug',
        drop_last=False,
        sampler=sampler,
        shuffle=False
    )
    if logger is not None:
        logger.debug(f"Test loader created with {len(loader)} batches of size {cfg.experiment.model.batch_size}.")
    
    return loader



####################################################################################################
############################################ INRIA #################################################
####################################################################################################
def get_train_loader_inria(cfg,tokenizer,logger=None):
    from .inria_coco import InriaCocoDatasetTrain, collate_fn

    ### ORIGINAL Pix2Poly Augmentations
    ## in my experiment this does not perform any better than D4+ColorJitter+GaussNoise
    train_transforms = A.Compose(
        [
            A.Affine(rotate=[-360, 360], fit_output=True, p=0.8),  # scaled rotations are performed before resizing to ensure rotated and scaled images are correctly resized.
            A.Resize(height=cfg.experiment.encoder.input_height, width=cfg.experiment.encoder.input_width),
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
        
    train_ds = InriaCocoDatasetTrain(
        cfg=cfg,
        transform=train_transforms,
        tokenizer=tokenizer,
    )
    
    sampler = DistributedSampler(dataset=train_ds, shuffle=True) if cfg.host.multi_gpu else None


    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.experiment.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.experiment.model.tokenizer.max_len, pad_idx=cfg.experiment.model.tokenizer.pad_idx),
        num_workers=cfg.num_workers,
        pin_memory=cfg.run_type.name!='debug',
        drop_last=True,
        sampler=sampler,
        shuffle=(sampler is None)
    )

    
    return train_loader

def get_val_loader_inria(cfg,tokenizer,logger=None):
    
    from .inria_coco import InriaCocoDatasetVal, collate_fn
    
    val_transforms = A.Compose(
        [
            A.Resize(height=cfg.experiment.encoder.input_height, width=cfg.experiment.encoder.input_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )
    
    
    val_ds = InriaCocoDatasetVal(
        cfg=cfg,
        transform=val_transforms,
        tokenizer=tokenizer,
    )
    
    if cfg.experiment.dataset.subset is not None:
        indices = list(range(cfg.experiment.dataset.subset))
        val_ds = Subset(val_ds, indices)
    
    ## be aware that the DistributedSampler here will even out the batch sizes on the different devices
    ## and thereby lead to some images being included twice in the coco annotations
    ## there is not really anything to avoid this, beside setting drop_last=True. Then, however, some images might be dropped entirely
    sampler = DistributedSampler(dataset=val_ds, shuffle=False) if cfg.host.multi_gpu else None

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.experiment.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.experiment.model.tokenizer.max_len, pad_idx=cfg.experiment.model.tokenizer.pad_idx),
        num_workers=cfg.num_workers,
        pin_memory=cfg.run_type.name!='debug',
        drop_last=False,
        sampler=sampler,
        shuffle=False
    )
    
    return val_loader



