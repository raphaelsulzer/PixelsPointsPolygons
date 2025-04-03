import numpy as np


def get_seg_display(seg):
    seg_display = np.zeros([seg.shape[0], seg.shape[1], 4], dtype=float)
    if len(seg.shape) == 2:
        seg_display[..., 0] = seg
        seg_display[..., 3] = seg
    else:
        for i in range(seg.shape[-1]):
            seg_display[..., i] = seg[..., i]
        seg_display[..., 3] = np.clip(np.sum(seg, axis=-1), 0, 1)
    return seg_display


def batch_to_cuda(batch):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cuda"):
            batch[key] = item.cuda(non_blocking=True)
    return batch


def batch_to_cpu(batch):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cpu"):
            batch[key] = item.cpu()
    return batch
