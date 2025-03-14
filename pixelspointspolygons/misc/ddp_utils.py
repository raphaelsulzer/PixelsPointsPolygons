import os
import torch
from torch import distributed as dist

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


def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
    dist_url = "env://"  # default

    # only works with torch.distributed.launch or torch.run.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # this will make all .cuda() calls work properly.
    torch.cuda.set_device(local_rank)

    # synchronizes all threads to reach this point before moving on.
    dist.barrier()
    
    return dist.get_world_size()