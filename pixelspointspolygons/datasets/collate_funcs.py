import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

def collate_fn_ffl(batch, cfg):
    
    batch_dict = defaultdict(list)
    
    for sample in batch:
        for key,val in sample.items():
            batch_dict[key].append(val)
            
    if cfg.use_images:
        assert (len(batch_dict["image"]) > 0), "Image batch is empty"
        batch_dict["image"] = torch.stack(batch_dict["image"])
    else:
        del batch_dict["image"]
    
    if cfg.use_lidar:
        assert (len(batch_dict["lidar"]) > 0), "LiDAR batch is empty"
        batch_dict["lidar"] = torch.nested.nested_tensor(batch_dict["lidar"], layout=torch.jagged)
    # else:
    #     batch_dict["lidar"] = None
    
    batch_dict["image_id"] = torch.stack(batch_dict["image_id"])
    batch_dict["distances"] = torch.stack(batch_dict["distances"])
    batch_dict["sizes"] = torch.stack(batch_dict["sizes"])
    batch_dict["gt_crossfield_angle"] = torch.stack(batch_dict["gt_crossfield_angle"])
    batch_dict["gt_polygons_image"] = torch.stack(batch_dict["gt_polygons_image"])
    batch_dict["class_freq"] = torch.stack(batch_dict["class_freq"])
    
    return dict(batch_dict)



def collate_fn_hisup(batch, cfg):
    
    image_batch, lidar_batch, annotations_batch, tile_id_batch = [], [], [], []
    for image, lidar, ann, tile_id in batch:
        if cfg.use_images:
            image_batch.append(image)
        if cfg.use_lidar:
            # lidar_pcd_id = torch.full((len(lidar),), i, dtype=torch.long)
            # lidar_pcd_id_batch.append(lidar_pcd_id)
            lidar_batch.append(lidar)
        
        annotations_batch.append(ann)
        tile_id_batch.append(tile_id)

    if cfg.use_images:
        image_batch = torch.stack(image_batch)
    else:
        image_batch = None
        
    if cfg.use_lidar:
        lidar_batch = torch.nested.nested_tensor(lidar_batch, layout=torch.jagged)
    else:
        lidar_batch = None
        
    # annotations_batch = torch.stack(annotations_batch)
    tile_id_batch = torch.stack(tile_id_batch)
    
    return image_batch, lidar_batch, annotations_batch, tile_id_batch




def collate_fn_pix2poly(batch, cfg):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    pad_idx = cfg.model.tokenizer.pad_idx
    max_len = cfg.model.tokenizer.max_len
    
    image_batch, lidar_batch, lidar_pcd_id_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, tile_id_batch = [], [], [], [], [], [], [], []
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
        tile_id_batch.append(idx)

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
    else:
        image_batch = None
        
    if cfg.use_lidar:
        # lidar_batch = torch.cat(lidar_batch)
        # lidar_pcd_id_batch = torch.cat(lidar_pcd_id_batch)
        ### try nested tensor instead of manuel indexing
        lidar_batch = torch.nested.nested_tensor(lidar_batch, layout=torch.jagged)
    else:
        lidar_batch = None
        
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    tile_id_batch = torch.stack(tile_id_batch)
    
    return image_batch, lidar_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, tile_id_batch