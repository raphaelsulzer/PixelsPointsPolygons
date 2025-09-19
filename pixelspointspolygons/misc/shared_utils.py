import sys
import os
import random
import torch
import contextlib

import numpy as np
import torch.distributed as dist

from copy import deepcopy
from collections import deque, OrderedDict
from omegaconf import OmegaConf

def get_experiment_type(experiment):
    name = experiment.split('/')
    if len(name) == 3:
        img_dim, polygonization_method, name = name
    elif len(name) == 2:
        img_dim, name = name
        polygonization_method = ""
    
    return name, img_dim, polygonization_method

def parse_cli_overrides():
    # Skip the script name
    return [arg for arg in sys.argv[1:] if "=" in arg]

def setup_hydraconf(cfg=None):
    """Setup OmegaConf to allow for dot notation and auto-completion"""

    OmegaConf.register_new_resolver("eq", lambda a, b: str(a) == str(b))
    OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond == "True" else b)
    OmegaConf.register_new_resolver("divide", lambda a, b: int(a) // int(b))
    if cfg is not None:
        OmegaConf.resolve(cfg)


@contextlib.contextmanager
def suppress_stdout():
    """Using this for surpressing the pycocotools messages"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def denormalize_image_for_visualization(image, cfg):
    """Reverse albumentations formula given here: https://explore.albumentations.ai/transform/Normalize
    A bit overkill with the division by image_max_pixel_value, but this is what is applied internally, so it should be checked here by showing that the plot is correct"""
    
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    std = np.array(cfg.experiment.encoder.image_std)
    mean = np.array(cfg.experiment.encoder.image_mean)
    max_pixel_val = cfg.experiment.encoder.image_max_pixel_value
    
    image = image * std * max_pixel_val + (mean * max_pixel_val)
    
    image = image.astype(np.uint8)
    
    return image

def smart_load_state_dict(model, checkpoint_state_dict, logger, strict=True):
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    unmatched_model_keys = set(model_state_dict.keys())
    unmatched_checkpoint_keys = set(checkpoint_state_dict.keys())

    temp_state_dict = deepcopy(checkpoint_state_dict)
    for k,v in checkpoint_state_dict.items():
        # temp_state_dict[k.replace(".vision_transformer.", ".vit.")] = v
        temp_state_dict[k.replace("encoder.model.", "encoder.vit.")] = v
    checkpoint_state_dict = temp_state_dict
    del temp_state_dict
    
    for ckpt_key in checkpoint_state_dict.keys():
        # Exact match first
        if ckpt_key in model_state_dict:
            new_state_dict[ckpt_key] = checkpoint_state_dict[ckpt_key]
            unmatched_model_keys.discard(ckpt_key)
            unmatched_checkpoint_keys.discard(ckpt_key)
        else:
            # Try to match by suffix (removing "module." or adding it)
            for model_key in model_state_dict.keys():
                if ckpt_key.endswith(model_key):
                    new_state_dict[model_key] = checkpoint_state_dict[ckpt_key]
                    unmatched_model_keys.discard(model_key)
                    unmatched_checkpoint_keys.discard(ckpt_key)
                    break
                if model_key.endswith(ckpt_key):
                    new_state_dict[model_key] = checkpoint_state_dict[ckpt_key]
                    unmatched_model_keys.discard(model_key)
                    unmatched_checkpoint_keys.discard(ckpt_key)
                    break
                    # Check for matches using the similar names dictionary
            
    logger.debug("Loading model state dict report")
    logger.debug(f"Matched {len(model_state_dict) - len(unmatched_model_keys)} / {len(model_state_dict)} keys")

    if unmatched_model_keys:
        logger.debug("Unmatched model keys (not found in checkpoint):")
        for k in unmatched_model_keys:
            logger.debug(f"  - {k}")

    if unmatched_checkpoint_keys:
        logger.debug("Unused checkpoint keys (not used in model):")
        for k in unmatched_checkpoint_keys:
            logger.debug(f"  - {k}")

    # Load matched params only
    model.load_state_dict(new_state_dict, strict=strict)

    return model

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.global_avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.global_avg = self.sum / self.count

    def __repr__(self) -> str:
        text = f"{self.name}: {self.global_avg:.4f}"
        return text

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_tile_names_from_dataloader(loader, ids):
    names = []
    # img_infos = loader.dataset.coco.loadImgs(ids)
    img_infos = np.array(list(loader.dataset.coco.imgs.values()))[ids]
    for i in range(len(ids)):
        names.append(img_infos[i]["file_name"].split("/")[-1].split(".")[0])
    return names

def plot_model_architecture(model, input_shape=(16,3,224,224), outfile="/data/rsulzer/model_architecture.svg"):
    from torchview import draw_graph

    model_graph = draw_graph(
        model,
        input_size=input_shape,  # adjust input shape
        expand_nested=True,
        graph_dir="LR",  # left-to-right layout
        save_graph=True,
        filename=outfile,
        # graph_attr={"dpi": "500"}
    )
    model_graph.visual_graph.render()  # render the graph to a file
    
    
def setup_ddp(cfg):
    """Init multi-gpu training or prediction"""
    
    if not cfg.host.multi_gpu:
        return 0,1
    else:
        world_size = torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])
    
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
        dist_url = "env://"  # default

        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=int(os.environ["RANK"])
        )
        
        # this will make all .cuda() calls work properly.
        torch.cuda.set_device(local_rank)

        # synchronizes all threads to reach this point before moving on.
        dist.barrier()
        
        return local_rank, world_size
    
    
def to_device(data,device):
    if isinstance(data,torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
#         import pdb; pdb.set_trace()
        for key in data:
            if isinstance(data[key],torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data,list):
        return [to_device(d,device) for d in data]

def to_single_device(data,device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    
    
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)