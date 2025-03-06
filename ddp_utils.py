from functools import partial
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch

from datasets.dataset_inria_coco import InriaCocoDataset, InriaCocoDatasetTest
from datasets.dataset_spacenet_coco import SpacenetCocoDataset, SpacenetCocoDatasetTest
from datasets.dataset_whu_buildings_coco import WHUBuildingsCocoDataset, WHUBuildingsCocoDatasetTest
from datasets.dataset_mass_roads import MassRoadsDataset, MassRoadsDatasetTest

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def collate_fn(batch, max_len, pad_idx):
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


# def collate_fn(batch, max_len, pad_idx):
#     """
#     if max_len:
#         the sequences will all be padded to that length.
#     """

#     image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch = [], [], [], [], []
#     for image, mask, c_mask, seq, perm_mat in batch:
#         image_batch.append(image)
#         mask_batch.append(mask)
#         coords_mask_batch.append(c_mask)
#         coords_seq_batch.append(seq)
#         perm_matrix_batch.append(perm_mat)

#     coords_seq_batch = pad_sequence(
#         coords_seq_batch,
#         padding_value=pad_idx,
#         batch_first=True
#     )

#     if max_len:
#         pad = torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1)).fill_(pad_idx).long()
#         coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)

#     image_batch = torch.stack(image_batch)
#     mask_batch = torch.stack(mask_batch)
#     coords_mask_batch = torch.stack(coords_mask_batch)
#     perm_matrix_batch = torch.stack(perm_matrix_batch)
#     return image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch

def get_lidar_poly_loaders(
    train_dataset_dir,
    val_dataset_dir,
    test_images_dir,
    tokenizer,
    n_vertices,
    max_len,
    pad_idx,
    shuffle_tokens,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):

    from lidar_poly_dataset import TrainDataset, ValDataset

    train_ds = TrainDataset(
        dataset_dir=train_dataset_dir,
        transform=train_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens,
        n_polygon_vertices=n_vertices
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True
    )

    valid_ds = ValDataset(
        dataset_dir=val_dataset_dir,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens,
        n_polygon_vertices=n_vertices
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, valid_loader, None




def get_inria_loaders(
    train_dataset_dir,
    val_dataset_dir,
    test_images_dir,
    tokenizer,
    max_len,
    pad_idx,
    shuffle_tokens,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True
):

    train_ds = InriaCocoDataset(
        dataset_dir=train_dataset_dir,
        transform=train_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    # train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        # sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_ds = InriaCocoDataset(
        dataset_dir=val_dataset_dir,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    # valid_sampler = DistributedSampler(dataset=valid_ds, shuffle=False)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        # sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )

    test_ds = InriaCocoDatasetTest(
        image_dir=test_images_dir,
        transform=val_transform
    )

    # test_sampler = DistributedSampler(dataset=test_ds, shuffle=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        # sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def get_spacenet_loaders(
    train_dataset_dir,
    val_dataset_dir,
    test_images_dir,
    tokenizer,
    max_len,
    pad_idx,
    shuffle_tokens,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True
):

    train_ds = SpacenetCocoDataset(
        dataset_dir=train_dataset_dir,
        transform=train_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_ds = SpacenetCocoDataset(
        dataset_dir=val_dataset_dir,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    valid_sampler = DistributedSampler(dataset=valid_ds, shuffle=False)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )

    test_ds = SpacenetCocoDatasetTest(
        image_dir=test_images_dir,
        transform=val_transform
    )

    test_sampler = DistributedSampler(dataset=test_ds, shuffle=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def get_whu_buildings_loaders(
    train_dataset_dir,
    val_dataset_dir,
    test_images_dir,
    tokenizer,
    max_len,
    pad_idx,
    shuffle_tokens,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True
):

    train_ds = WHUBuildingsCocoDataset(
        dataset_dir=train_dataset_dir,
        transform=train_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_ds = WHUBuildingsCocoDataset(
        dataset_dir=val_dataset_dir,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    valid_sampler = DistributedSampler(dataset=valid_ds, shuffle=False)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )

    test_ds = WHUBuildingsCocoDatasetTest(
        image_dir=test_images_dir,
        transform=val_transform
    )

    test_sampler = DistributedSampler(dataset=test_ds, shuffle=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def get_mass_roads_loaders(
    train_dataset_dir,
    val_dataset_dir,
    test_images_dir,
    tokenizer,
    max_len,
    pad_idx,
    shuffle_tokens,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True
):

    train_ds = MassRoadsDataset(
        dataset_dir=train_dataset_dir,
        transform=train_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_ds = MassRoadsDataset(
        dataset_dir=val_dataset_dir,
        transform=val_transform,
        tokenizer=tokenizer,
        shuffle_tokens=shuffle_tokens
    )

    valid_sampler = DistributedSampler(dataset=valid_ds, shuffle=False)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )

    test_ds = MassRoadsDatasetTest(
        image_dir=test_images_dir,
        transform=val_transform
    )

    test_sampler = DistributedSampler(dataset=test_ds, shuffle=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader

