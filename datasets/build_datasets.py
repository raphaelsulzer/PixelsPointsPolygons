import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from datasets.dataset_inria_coco import InriaCocoDatasetTrain, InriaCocoDatasetVal


def get_val_loader(cfg,tokenizer):
    if cfg.dataset.name == 'inria':
        return get_val_loader_inria(cfg,tokenizer)
    else:
        raise NotImplementedError

def get_val_loader_inria(cfg,tokenizer):
    
    from datasets.dataset_inria_coco import collate_fn
    
    valid_transforms = A.Compose(
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
        transform=valid_transforms,
        tokenizer=tokenizer,
    )
    
    if cfg.subset is not None:
        indices = list(range(cfg.subset))
        val_ds = Subset(val_ds, indices)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.model.batch_size,
        collate_fn=partial(collate_fn, max_len=cfg.model.tokenizer.max_len, pad_idx=cfg.model.tokenizer.pad_idx),
        num_workers=cfg.model.num_workers
    )
    
    return val_loader



