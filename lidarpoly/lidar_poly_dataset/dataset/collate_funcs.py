import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn_pix2poly(batch, cfg):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    pad_idx = cfg.model.tokenizer.pad_idx
    max_len = cfg.model.tokenizer.max_len
    
    image_batch, lidar_batch, lidar_pcd_id_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, image_id_batch = [], [], [], [], [], [], [], []
    for i, (image, lidar, mask, c_mask, seq, perm_mat, idx) in enumerate(batch):
        if cfg.use_images:
            image_batch.append(image)
        if cfg.use_lidar:
            # lidar_pcd_id = torch.full((len(lidar),), i, dtype=torch.long)
            # lidar_pcd_id_batch.append(lidar_pcd_id)
            lidar_batch.append(lidar)
        
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)
        image_id_batch.append(idx)

    coords_seq_batch = pad_sequence(
        coords_seq_batch,
        padding_value=pad_idx,
        batch_first=True
    )

    if max_len:
        pad = torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1)).fill_(pad_idx).long()
        coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)

    if cfg.use_images:
        image_batch = torch.stack(image_batch)
    if cfg.use_lidar:
        # lidar_batch = torch.cat(lidar_batch)
        # lidar_pcd_id_batch = torch.cat(lidar_pcd_id_batch)
        ### try nested tensor instead of manuel indexing
        lidar_batch = torch.nested.nested_tensor(lidar_batch, layout=torch.jagged)
        
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    image_id_batch = torch.stack(image_id_batch)
    return image_batch, lidar_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, image_id_batch