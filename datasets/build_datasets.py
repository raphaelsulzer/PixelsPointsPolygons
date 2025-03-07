import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence


def get_val_loader(cfg,tokenizer):
    if cfg.dataset.name == 'inria':
        return get_val_loader_inria(cfg,tokenizer)
    elif cfg.dataset.name == 'lidarpoly':
        return get_val_loader_lidarpoly(cfg,tokenizer)
    else:
        raise NotImplementedError


def collate_fn_lidarpoly(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, idx_batch = [], [], [], [], [], []
    for image, mask, c_mask, seq, perm_mat, idx in batch:
        image_batch.append(image)
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)
        idx_batch.append(idx)

    coords_seq_batch = pad_sequence(
        coords_seq_batch,
        padding_value=pad_idx,
        batch_first=True
    )

    if max_len:
        pad = torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1)).fill_(pad_idx).long()
        coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)

    image_batch = torch.stack(image_batch)
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    idx_batch = torch.stack(idx_batch)
    return image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, idx_batch

def get_val_loader_lidarpoly(cfg,tokenizer):
    
    
    from lidar_poly_dataset import TrainDataset, ValDataset
    
    val_transform = A.Compose(
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
        
    val_ds = ValDataset(
        dataset_dir=cfg.dataset.path,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=cfg.model.tokenizer.shuffle_tokens,
        n_polygon_vertices=cfg.model.tokenizer.n_vertices
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn_lidarpoly, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        num_workers=cfg.num_workers
    )
    
    return val_loader


def get_val_loader_inria(cfg,tokenizer):
    
    from datasets.dataset_inria_coco import InriaCocoDatasetTrain, InriaCocoDatasetVal, collate_fn
    
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
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        num_workers=cfg.num_workers
    )
    
    return val_loader



