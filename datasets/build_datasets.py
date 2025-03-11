import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
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
    
    from lidar_poly_dataset import TrainDataset
    from datasets.dataset_inria_coco import collate_fn

    train_transforms = A.ReplayCompose([
        # A.D4(p=1.0),
        A.Resize(height=cfg.model.input_height, width=cfg.model.input_width),
        A.ColorJitter(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format='yx')
    )
    
    train_ds = TrainDataset(
        dataset_dir=cfg.dataset.path,
        transform=train_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=cfg.model.tokenizer.shuffle_tokens,
        n_polygon_vertices=cfg.model.tokenizer.n_vertices
    )

    if cfg.multi_gpu:
        train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return train_loader

def get_val_loader_lidarpoly(cfg,tokenizer):
    
    
    from lidar_poly_dataset import ValDataset
    from datasets.dataset_inria_coco import collate_fn

    
    val_transforms = A.Compose(
        [
            A.Resize(height=cfg.model.input_height, width=cfg.model.input_width),
            A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )
        
    val_ds = ValDataset(
        dataset_dir=cfg.dataset.path,
        transform=val_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=cfg.model.tokenizer.shuffle_tokens,
        n_polygon_vertices=cfg.model.tokenizer.n_vertices
    )

    if cfg.multi_gpu:
        val_sampler = DistributedSampler(dataset=val_ds, shuffle=False)
    else:
        val_sampler = None
        
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return val_loader



def get_train_loader_inria(cfg,tokenizer):
    from datasets.dataset_inria_coco import InriaCocoDatasetTrain, collate_fn

    train_transforms = A.ReplayCompose([
        # A.D4(p=1.0),
        A.Resize(height=cfg.model.input_height, width=cfg.model.input_width),
        A.ColorJitter(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format='yx')
    )
        
    train_ds = InriaCocoDatasetTrain(
        cfg=cfg,
        transform=train_transforms,
        tokenizer=tokenizer,
    )
    if cfg.multi_gpu:
        train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
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
        val_sampler = DistributedSampler(dataset=val_ds, shuffle=False)
    else:
        val_sampler = None
        
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return val_loader



