import sys
import os
import random
import torch
import contextlib

import numpy as np

from collections import deque
from scipy.optimize import linear_sum_assignment
from transformers.generation.utils import top_k_top_p_filtering

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


    
def compute_dynamic_cfg_vars(cfg,tokenizer):
    
    cfg.model.tokenizer.pad_idx = tokenizer.PAD_code
    cfg.model.tokenizer.max_len = cfg.model.tokenizer.n_vertices*2+2
    cfg.model.tokenizer.generation_steps = cfg.model.tokenizer.n_vertices*2+1
    cfg.model.encoder.num_patches = int((cfg.model.encoder.input_size // cfg.model.encoder.patch_size) ** 2)
    

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_square_subsequent_mask(sz,device):
    mask = (
        torch.triu(torch.ones((sz, sz), device=device)) == 1
    ).transpose(0, 1)

    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

    return mask


def create_mask(tgt, pad_idx):
    """
    tgt shape: (N, L)
    """

    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device=tgt.device)
    # changing the type here from bool to float32 to get rid of the torch warning
    tgt_padding_mask = (tgt == pad_idx).to(dtype=tgt_mask.dtype)

    return tgt_mask, tgt_padding_mask


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


def scores_to_permutations(scores):
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        r, c = linear_sum_assignment(-scores[b])
        perm[b,r,c] = 1
    return torch.tensor(perm)


# TODO: add permalink to polyworld repo
def permutations_to_polygons(perm, graph, out='torch'):
    B, N, N = perm.shape
    device = perm.device

    def bubble_merge(poly):
        s = 0
        P = len(poly)
        while s < P:
            head = poly[s][-1]

            t = s+1
            while t < P:
                tail = poly[t][0]
                if head == tail:
                    poly[s] = poly[s] + poly[t][1:]
                    del poly[t]
                    poly = bubble_merge(poly)
                    P = len(poly)
                t += 1
            s += 1
        return poly

    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b]
        b_diag = diag[b]

        idx = torch.arange(N, device=perm.device)[b_diag]

        if idx.shape[0] > 0:
            # If there are vertices in the batch

            b_perm = b_perm[idx,:]
            b_graph = b_graph[idx,:]
            b_perm = b_perm[:,idx]

            first = torch.arange(idx.shape[0]).unsqueeze(1).to(device=device)
            second = torch.argmax(b_perm, dim=1).unsqueeze(1)

            polygons_idx = torch.cat((first, second), dim=1).tolist()
            polygons_idx = bubble_merge(polygons_idx)

            batch_poly = []
            for p_idx in polygons_idx:
                if out == 'torch':
                    batch_poly.append(b_graph[p_idx,:])
                elif out == 'numpy':
                    batch_poly.append(b_graph[p_idx,:].cpu().numpy())
                elif out == 'list':
                    g = b_graph[p_idx,:] * 300 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist())
                elif out == 'coco':
                    g = b_graph[p_idx,:]
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist())
                elif out == 'inria-torch':
                    batch_poly.append(b_graph[p_idx,:])
                else:
                    print("Indicate a valid output polygon format")
                    exit()

            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch


def test_generate(model, x_images, x_lidar, tokenizer, max_len=50, top_k=0, top_p=1):
    
    batch_size = x_images.size(0) if x_images is not None else x_lidar.size(0)
    device = x_images.device if x_images is not None else x_lidar.device
    
    batch_preds = torch.ones((batch_size, 1), device=device).fill_(tokenizer.BOS_code).long()

    confs = []

    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

    with torch.no_grad():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            encoded_image = model.module.encoder(x_images, x_lidar)
        else:
            encoded_image = model.encoder(x_images, x_lidar)
        for i in range(max_len):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                preds, feats = model.module.predict(encoded_image, batch_preds)
            else:
                preds, feats = model.predict(encoded_image, batch_preds)
                
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)  # if top_k and top_p are set to default, this line does nothing.
            if i % 2 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            perm_preds = model.module.scorenet1(feats) + torch.transpose(model.module.scorenet2(feats), 1, 2)
        else:
            perm_preds = model.scorenet1(feats) + torch.transpose(model.scorenet2(feats), 1, 2)

        perm_preds = scores_to_permutations(perm_preds)

    return batch_preds.cpu(), confs, perm_preds


def postprocess(batch_preds, batch_confs, tokenizer):
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 2 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0

    all_coords = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_coords.append(None)
            all_confs.append(None)
            continue
        coords = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(coords))]

        all_coords.append(coords)
        all_confs.append(confs)

    return all_coords, all_confs


def get_tile_names_from_dataloader(img_dict, ids):
    file_names = []
    for id in ids:
        file_names.append(img_dict[id.item()]['file_name'])
    return file_names


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