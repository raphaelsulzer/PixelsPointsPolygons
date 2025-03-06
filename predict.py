import os
import time
import json
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

import torch
from torchvision.utils import make_grid
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from config import CFG
from tokenizer import Tokenizer
from utils import (
    seed_everything,
    test_generate_arno,
    postprocess,
    permutations_to_polygons,
)
from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder,
    EncoderDecoderWithAlreadyEncodedImages,
)
from datasets.build_datasets import get_val_loader

from lidar_poly_dataset.utils import generate_coco_ann


def get_model(cfg,tokenizer):
    
    cfg.model.tokenizer.max_len = cfg.model.tokenizer.n_vertices*2+2
    cfg.model.num_patches = int((cfg.model.input_size // cfg.model.patch_size) ** 2)
    
    encoder = Encoder(model_name=cfg.model.type, pretrained=True, out_dim=256)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        encoder_len=cfg.model.num_patches,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_len=cfg.model.tokenizer.max_len,
        pad_idx=cfg.model.tokenizer.pad_idx,
    )
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        n_vertices=cfg.model.tokenizer.n_vertices,
        sinkhorn_iterations=CFG.SINKHORN_ITERATIONS,
    )
    model.to(cfg.device)
    model.eval()
    model_taking_encoded_images = EncoderDecoderWithAlreadyEncodedImages(model)
    model_taking_encoded_images.to(cfg.device)
    model_taking_encoded_images.eval()
    
    return model, model_taking_encoded_images


def make_pixel_mask_from_prediction(x,batch_polygons,cfg):
    B, C, H, W = x.shape

    polygons_mask = np.zeros((B, 1, H, W))
    for b in range(len(batch_polygons)):
        for c in range(len(batch_polygons[b])):
            poly = batch_polygons[b][c]
            poly = poly[poly[:, 0] != cfg.model.tokenizer.pad_idx]
            cnt = np.flip(np.int32(poly.cpu()), 1)
            if len(cnt) > 0:
                cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.)
    return torch.from_numpy(polygons_mask)
    

def plot_predictions(polygon_mask, y_mask, outfile):

    pred_grid = make_grid(polygon_mask).permute(1, 2, 0)
    gt_grid = make_grid(y_mask).permute(1, 2, 0)
    plt.subplot(211), plt.imshow(pred_grid) ,plt.title("Predicted Polygons") ,plt.axis('off')
    plt.subplot(212), plt.imshow(gt_grid) ,plt.title("Ground Truth") ,plt.axis('off')
    
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def run_prediction(cfg):
    seed_everything(42)

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=cfg.model.tokenizer.num_bins,
        width=cfg.model.input_width,
        height=cfg.model.input_height,
        max_len=cfg.model.tokenizer.max_len
    )
    cfg.model.tokenizer.pad_idx = tokenizer.PAD_code

    val_loader = get_val_loader(cfg,tokenizer)
    model, model_taking_encoded_images = get_model(cfg,tokenizer)

    checkpoint = torch.load(cfg.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epochs_run']

    print(f"Model loaded from epoch: {epoch}")

    mean_iou_metric = BinaryJaccardIndex()
    mean_acc_metric = BinaryAccuracy()


    with torch.no_grad():
        cumulative_miou = []
        cumulative_macc = []
        speed = []
        predictions = []
        for i_batch, (x, y_mask, y_corner_mask, y, y_perm, idx) in enumerate(tqdm(val_loader)):
            all_coords = []
            all_confs = []
            t0 = time.time()
            # batch_preds, batch_confs, perm_preds = test_generate(model.encoder, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
            batch_preds, batch_confs, perm_preds = test_generate_arno(
                model.encoder,
                model_taking_encoded_images,
                x,
                tokenizer,
                max_len=CFG.generation_steps,
                top_k=0,
                top_p=1,
            )
            
            speed.append(time.time() - t0)
            vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

            all_coords.extend(vertex_coords)
            all_confs.extend(confs)

            coords = []
            for i in range(len(all_coords)):
                if all_coords[i] is not None:
                    coord = torch.from_numpy(all_coords[i])
                else:
                    coord = torch.tensor([])

                padd = torch.ones((cfg.model.tokenizer.n_vertices - len(coord), 2)).fill_(cfg.model.tokenizer.pad_idx)
                coord = torch.cat([coord, padd], dim=0)
                coords.append(coord)
            batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]     

            for ip, pp in enumerate(batch_polygons):
                polys = []
                for p in pp:
                    p = torch.fliplr(p)
                    p = p[p[:, 0] != cfg.model.tokenizer.pad_idx]
                    polys.append(p)
                predictions.extend(generate_coco_ann(polys,idx[ip].item()))

            polygons_mask = make_pixel_mask_from_prediction(x,batch_polygons,cfg)
            batch_miou = mean_iou_metric(polygons_mask, y_mask)
            batch_macc = mean_acc_metric(polygons_mask, y_mask)
            # outfile = os.path.join(cfg.output_dir, "images", f"predictions_{i_batch}.png")
            # plot_predictions(polygons_mask, y_mask, outfile)

            cumulative_miou.append(batch_miou)
            cumulative_macc.append(batch_macc)

        print("Average model speed: ", np.mean(speed) / cfg.model.batch_size, " [s / image]")

        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}")
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}")

    checkpoint_name = os.path.split(cfg.checkpoint)[-1][:-4]
    prediction_outfile = os.path.join(cfg.output_dir, f"predictions_{checkpoint_name}.json")
    with open(prediction_outfile, "w") as fp:
        fp.write(json.dumps(predictions))

    eval_outfile = os.path.join(cfg.output_dir, f"metrics_{checkpoint_name}.json")
    with open(eval_outfile, 'w') as ff:
        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}", file=ff)
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}", file=ff)



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    run_prediction(cfg)

if __name__ == "__main__":
    main()
