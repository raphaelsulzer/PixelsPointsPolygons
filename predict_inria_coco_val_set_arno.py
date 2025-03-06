import argparse
import json
import os
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
from torchvision.utils import make_grid
from tqdm import tqdm

from config_arno import Pix2PolyConfig
from datasets.dataset_inria_coco import InriaCocoDataset_val
from models.model_ori import (
    Decoder,
    Encoder,
    EncoderDecoder,
    EncoderDecoderWithAlreadyEncodedImages,
)
from tokenizer import Tokenizer
from utils import (
    permutations_to_polygons,
    postprocess,
    seed_everything,
    test_generate,
)


def bounding_box_from_points(points: List[float]) -> List[int]:
    pts = np.array(points).flatten()
    even_locations = np.arange(len(pts) // 2) * 2
    odd_locations = even_locations + 1
    X = np.take(pts, even_locations.tolist())
    Y = np.take(pts, odd_locations.tolist())
    bbox = [int(X.min()), int(Y.min()), int(X.max() - X.min()), int(Y.max() - Y.min())]
    return bbox


def single_annotation(image_id: int, poly: List[List[float]]) -> Dict[str, Any]:
    result = {}
    result["image_id"] = int(image_id)
    result["category_id"] = 100
    result["score"] = 1
    result["segmentation"] = poly
    result["bbox"] = bounding_box_from_points(result["segmentation"])
    return result


def collate_fn(
    batch: List[Tuple[torch.Tensor, ...]], max_len: int, pad_idx: int
) -> Tuple[torch.Tensor, ...]:
    (
        image_batch,
        mask_batch,
        coords_mask_batch,
        coords_seq_batch,
        perm_matrix_batch,
        idx_batch,
    ) = [], [], [], [], [], []
    for image, mask, c_mask, seq, perm_mat, idx in batch:
        image_batch.append(image)
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)
        idx_batch.append(idx)
    coords_seq_batch = pad_sequence(
        coords_seq_batch, padding_value=pad_idx, batch_first=True
    )
    if max_len:
        pad = (
            torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1))
            .fill_(pad_idx)
            .long()
        )
        coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    idx_batch = torch.stack(idx_batch)
    return (
        image_batch,
        mask_batch,
        coords_mask_batch,
        coords_seq_batch,
        perm_matrix_batch,
        idx_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-f",
        type=Path,
        default=Path(__file__).parent / "datasets" / "configs" / "inria.json",
        help="Path to config JSON file.",
        required=True,
    )
    parser.add_argument(
        "--override_checkpoint_path",
        "-c",
        type=Path,
        help="Choice of checkpoint to evaluate in experiment. If not provided, will use the one in the config file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Name of output subdirectory to store part predictions.",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")

    config = Pix2PolyConfig.parse_file(Path(args.config_file))
    checkpoint_path = (
        Path(args.override_checkpoint_path)
        if args.override_checkpoint_path is not None
        else config.checkpoint_path
    )
    run_name_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(args.output_dir) / run_name_string
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(42)

    valid_transforms = A.Compose(
        [
            A.Resize(height=config.input_height, width=config.input_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="yx", remove_invisible=False),
    )

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=config.num_bins,
        width=config.input_width,
        height=config.input_height,
        max_len=config.max_len,
    )
    config.pad_idx = tokenizer.PAD_code

    val_ds = InriaCocoDataset_val(
        dataset_dir=config.val_dataset_dir,
        n_vertices=config.n_vertices,
        transform=valid_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=config.shuffle_tokens,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        collate_fn=partial(collate_fn, max_len=config.max_len, pad_idx=config.pad_idx),
        num_workers=20,
        pin_memory=True,
        persistent_workers=True,
    )

    encoder = Encoder(model_name=config.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        encoder_len=config.num_patches,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_len=config.max_len,
        pad_idx=config.pad_idx,
    )
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        n_vertices=config.n_vertices,
        sinkhorn_iterations=config.sinkhorn_iterations,
    )
    model.to(config.device)
    model.eval()
    model_taking_encoded_images = EncoderDecoderWithAlreadyEncodedImages(model)
    model_taking_encoded_images.to(config.device)
    model_taking_encoded_images.eval()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint.get("epochs_run", 0)

    print(f"Model loaded from epoch: {epoch}")
    ckpt_desc = f"epoch_{epoch}"
    chkpt_basename = os.path.basename(checkpoint_path)
    if "best_valid_loss" in chkpt_basename:
        ckpt_desc = f"epoch_{epoch}_bestValLoss"
    elif "best_valid_metric" in chkpt_basename:
        ckpt_desc = f"epoch_{epoch}_bestValMetric"

    mean_iou_metric = BinaryJaccardIndex()
    mean_acc_metric = BinaryAccuracy()

    with torch.no_grad():
        cumulative_miou = []
        cumulative_macc = []
        speed = []
        predictions = []
        for i_batch, (x, y_mask, _, _, _, idx) in enumerate(tqdm(val_loader)):
            t0 = time.time()
            batch_preds, batch_confs, perm_preds = test_generate(
                model.encoder,
                model_taking_encoded_images,
                x,
                tokenizer,
                max_len=config.generation_steps,
                top_k=0,
                top_p=1,
            )
            speed.append(time.time() - t0)
            vertex_coords, _ = postprocess(batch_preds, batch_confs, tokenizer)

            coords = []
            for coord_data in vertex_coords:
                if coord_data is not None:
                    coord = torch.from_numpy(coord_data)
                else:
                    coord = torch.tensor([])
                pad_tensor = torch.ones((config.n_vertices - len(coord), 2)).fill_(
                    config.pad_idx
                )
                coord = torch.cat([coord, pad_tensor], dim=0)
                coords.append(coord)
            batch_polygons = permutations_to_polygons(perm_preds, coords, out="torch")
            for ip, polygon_list in enumerate(batch_polygons):
                for poly_tensor in polygon_list:
                    poly_tensor = torch.fliplr(poly_tensor)
                    poly_tensor = poly_tensor[poly_tensor[:, 0] != config.pad_idx]
                    poly_tensor = poly_tensor * (config.img_size / config.input_width)
                    poly_list = poly_tensor.view(-1).tolist()
                    if len(poly_list) > 0:
                        predictions.append(single_annotation(idx[ip], [poly_list]))

            B, _, H, W = x.shape
            polygons_mask = np.zeros((B, 1, H, W))
            for b in range(len(batch_polygons)):
                for poly_tensor in batch_polygons[b]:
                    poly_tensor = poly_tensor[poly_tensor[:, 0] != config.pad_idx]
                    cnt = np.flip(np.int32(poly_tensor.cpu()), 1)
                    if len(cnt) > 0:
                        cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.0)
            polygons_mask = torch.from_numpy(polygons_mask)

            batch_miou = mean_iou_metric(polygons_mask, y_mask)
            batch_macc = mean_acc_metric(polygons_mask, y_mask)

            cumulative_miou.append(batch_miou)
            cumulative_macc.append(batch_macc)

            pred_grid = make_grid(polygons_mask).permute(1, 2, 0)
            gt_grid = make_grid(y_mask).permute(1, 2, 0)
            (
                plt.subplot(211),
                plt.imshow(pred_grid),
                plt.title("Predicted Polygons"),
                plt.axis("off"),
            )
            (
                plt.subplot(212),
                plt.imshow(gt_grid),
                plt.title("Ground Truth"),
                plt.axis("off"),
            )
            date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            fig_dir = f"/home/agobbin@ms.luxcarta.com/garbage/figs/{date_string}"
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(f"{fig_dir}/batch_{i_batch}.png")
            plt.close()

        print(
            "Average model speed: ", np.mean(speed) / config.batch_size, " [s / image]"
        )
        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}")
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}")

    dataset_name = config.val_dataset_dir.name
    part_desc_name = "predictions"
    preds_path = (
        output_dir / f"predictions_{dataset_name}_{part_desc_name}_{ckpt_desc}.json"
    )
    with open(preds_path, "w") as fp:
        fp.write(json.dumps(predictions))

    metrics_path = (
        output_dir / f"val_metrics_{dataset_name}_{part_desc_name}_{ckpt_desc}.txt"
    )
    with open(metrics_path, "w") as ff:
        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}", file=ff)
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}", file=ff)


if __name__ == "__main__":
    main()