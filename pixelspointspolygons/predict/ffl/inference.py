import sys

from tqdm import tqdm
import scipy

import numpy as np
import torch

from ...models.ffl import local_utils
from . import polygonize

from lydorn_utils import image_utils
from lydorn_utils import print_utils
from lydorn_utils import python_utils


def network_inference(model, batch, device):
    batch = local_utils.batch_to_cuda(batch,device=device)
    pred = model(batch)
    return pred, batch


def inference(cfg, model, tile_data, compute_polygonization=False, pool=None):
    if cfg.model.eval_params.patch_size is not None:
        # Cut image into patches for inference
        inference_with_patching(cfg, model, tile_data)
        single_sample = True
    else:
        # Feed images as-is to the model
        inference_no_patching(cfg, model, tile_data)
        single_sample = False

    # Polygonize:
    if compute_polygonization:
        pool = None if single_sample else pool  # A single big image is being processed
        crossfield = tile_data["crossfield"] if "crossfield" in tile_data else None
        polygons_batch, probs_batch = polygonize.polygonize(cfg["polygonize_params"], tile_data["seg"], crossfield_batch=crossfield,
                                         pool=pool)
        tile_data["polygons"] = polygons_batch
        tile_data["polygon_probs"] = probs_batch

    return tile_data


def inference_no_patching(cfg, model, batch):
    with torch.no_grad():

        pred, batch = network_inference(model, batch, device=cfg.host.device)
        
        batch["seg"] = pred["seg"]
        if "crossfield" in pred:
            batch["crossfield"] = pred["crossfield"]

    return batch


def inference_with_patching(cfg, model, tile_data):
    assert len(tile_data["image"].shape) == 4 and tile_data["image"].shape[0] == 1, \
        f"When using inference with patching, tile_data should have a batch size of 1, " \
        f"with image's shape being (1, C, H, W), not {tile_data['image'].shape}"
    with torch.no_grad():
        # Init tile outputs (image is (N, C, H, W)):
        height = tile_data["image"].shape[2]
        width = tile_data["image"].shape[3]
        seg_channels = cfg["seg_params"]["compute_interior"] \
                       + cfg["seg_params"]["compute_edge"] \
                       + cfg["seg_params"]["compute_vertex"]
        if cfg["compute_seg"]:
            tile_data["seg"] = torch.zeros((1, seg_channels, height, width), device=cfg["device"])
        if cfg["compute_crossfield"]:
            tile_data["crossfield"] = torch.zeros((1, 4, height, width), device=cfg["device"])
        weight_map = torch.zeros((1, 1, height, width), device=cfg["device"])  # Count number of patches on top of each pixel

        # Split tile in patches:
        stride = cfg["eval_params"]["patch_size"] - cfg["eval_params"]["patch_overlap"]
        patch_boundingboxes = image_utils.compute_patch_boundingboxes((height, width),
                                                                      stride=stride,
                                                                      patch_res=cfg["eval_params"]["patch_size"])
        # Compute patch pixel weights to merge overlapping patches back together smoothly:
        patch_weights = np.ones((cfg["eval_params"]["patch_size"] + 2, cfg["eval_params"]["patch_size"] + 2),
                                dtype=np.float)
        patch_weights[0, :] = 0
        patch_weights[-1, :] = 0
        patch_weights[:, 0] = 0
        patch_weights[:, -1] = 0
        patch_weights = scipy.ndimage.distance_transform_edt(patch_weights)
        patch_weights = patch_weights[1:-1, 1:-1]
        patch_weights = torch.tensor(patch_weights, device=cfg["device"]).float()
        patch_weights = patch_weights[None, None, :, :]  # Adding batch and channels dims

        # Predict on each patch and save in outputs:
        for bbox in tqdm(patch_boundingboxes, desc="Running model on patches", leave=False):
            # Crop data
            batch = {
                "image": tile_data["image"][:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]],
                "image_mean": tile_data["image_mean"],
                "image_std": tile_data["image_std"],
            }
            # Send batch to device
            try:
                pred, batch = network_inference(cfg, model, batch, device=cfg.host.device)
            except RuntimeError as e:
                print_utils.print_error("ERROR: " + str(e))
                print_utils.print_info("INFO: Reduce --eval_patch_size until the patch fits in memory.")
                raise e

            if cfg["compute_seg"]:
                tile_data["seg"][:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights * pred["seg"]
            if cfg["compute_crossfield"]:
                tile_data["crossfield"][:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights * pred["crossfield"]
            weight_map[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] += patch_weights

        # Take care of overlapping parts
        if cfg["compute_seg"]:
            tile_data["seg"] /= weight_map
        if cfg["compute_crossfield"]:
            tile_data["crossfield"] /= weight_map

    return tile_data