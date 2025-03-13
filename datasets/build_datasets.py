import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler


def get_val_loader(cfg,tokenizer):
    if cfg.dataset.name == 'inria':
        return get_val_loader_inria(cfg,tokenizer)
    elif cfg.dataset.name == 'lidarpoly':
        return get_val_loader_lidarpoly(cfg,tokenizer)
    else:
        raise NotImplementedError

def get_train_loader(cfg,tokenizer):
    if cfg.dataset.name == 'inria':
        return get_train_loader_inria(cfg,tokenizer)
    elif cfg.dataset.name == 'lidarpoly':
        return get_train_loader_lidarpoly(cfg,tokenizer)
    else:
        raise NotImplementedError

def get_train_loader_lidarpoly(cfg,tokenizer):
    
    from lidar_poly_dataset.dataset import TrainDataset, collate_fn_pix2poly

    ### ORIGINAL
    train_transforms = A.Compose(
        [
            A.Affine(rotate=[-360, 360], fit_output=True, p=0.8),  # scaled rotations are performed before resizing to ensure rotated and scaled images are correctly resized.
            A.Resize(height=cfg.model.encoder.input_height, width=cfg.model.encoder.input_width),
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
    
    # train_transforms = A.ReplayCompose([
    #     A.D4(p=1.0),
    #     A.Resize(height=cfg.model.input_height, width=cfg.model.input_width),
    #     A.ColorJitter(),
    #     A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    #     ToTensorV2(),
    # ],
    #     keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    # )
    
    train_ds = TrainDataset(
        cfg,
        transform=train_transforms,
        tokenizer=tokenizer
    )

    if cfg.multi_gpu:
        train_sampler = {"sampler":DistributedSampler(dataset=train_ds, shuffle=True)}
    else:
        train_sampler = {"shuffle":True}

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn_pix2poly, cfg=cfg),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        **train_sampler
    )
    
    return train_loader

def get_val_loader_lidarpoly(cfg,tokenizer):
    
    
    from lidar_poly_dataset.dataset import ValDataset, collate_fn_pix2poly

    
    val_transforms = A.Compose(
        [
            A.Resize(height=cfg.model.encoder.input_height, width=cfg.model.encoder.input_width),
            A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )
    
    val_ds = ValDataset(
        cfg,
        transform=val_transforms,
        tokenizer=tokenizer
    )


    if cfg.multi_gpu:
        val_sampler = {"sampler":DistributedSampler(dataset=val_ds, shuffle=False)}
    else:
        val_sampler = {"shuffle":False}
        
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn_pix2poly, cfg=cfg),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        **val_sampler
    )
    
    return val_loader



def get_train_loader_inria(cfg,tokenizer):
    from datasets.dataset_inria_coco import InriaCocoDatasetTrain, collate_fn

    ### ORIGINAL
    train_transforms = A.Compose(
        [
            A.Affine(rotate=[-360, 360], fit_output=True, p=0.8),  # scaled rotations are performed before resizing to ensure rotated and scaled images are correctly resized.
            A.Resize(height=cfg.model.input_height, width=cfg.model.input_width),
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
        
    if cfg.multi_gpu:
        train_sampler = {"sampler":DistributedSampler(dataset=train_ds, shuffle=True)}
    else:
        train_sampler = {"shuffle":True}

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        **train_sampler
    )
    
    return train_loader

def get_val_loader_inria(cfg,tokenizer):
    
    from datasets.dataset_inria_coco import InriaCocoDatasetVal, collate_fn
    
    val_transforms = A.Compose(
        [
            A.Resize(height=cfg.model.input_height, width=cfg.model.input_width),
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
    
    if cfg.dataset.subset is not None:
        indices = list(range(cfg.dataset.subset))
        val_ds = Subset(val_ds, indices)
    
        
    if cfg.multi_gpu:
        val_sampler = {"sampler":DistributedSampler(dataset=val_ds, shuffle=True)}
    else:
        val_sampler = {"shuffle":True}
        
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        **val_sampler
    )
    
    return val_loader



