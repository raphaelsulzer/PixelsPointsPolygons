import sys
import os
import random
import torch
import contextlib

import numpy as np
import torch.distributed as dist

from collections import deque

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
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        text = f"{self.name}: {self.avg:.4f}"
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
    img_infos = loader.dataset.coco.loadImgs(ids)
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
    
    
def setup_ddp(world_size, local_rank):
    """Init multi-gpu training or prediction"""
    
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