import os
import time
import json
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

from functools import partial
import torch
from torch.utils.data import Subset

from torchvision.utils import make_grid


from config import CFG
from tokenizer import Tokenizer
from ddp_utils import collate_fn
from utils_ori import (
    seed_everything,
    load_checkpoint,
    test_generate,
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

from torch.utils.data import DataLoader
from datasets.dataset_inria_coco import InriaCocoDataset_val
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from lidar_poly_dataset.utils import generate_coco_ann


def get_model(tokenizer):
    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_len=CFG.MAX_LEN,
        pad_idx=CFG.PAD_IDX,
    )
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        n_vertices=CFG.N_VERTICES,
        sinkhorn_iterations=CFG.SINKHORN_ITERATIONS,
    )
    model.to(CFG.DEVICE)
    model.eval()
    model_taking_encoded_images = EncoderDecoderWithAlreadyEncodedImages(model)
    model_taking_encoded_images.to(CFG.DEVICE)
    model_taking_encoded_images.eval()
    
    return model, model_taking_encoded_images


def run_prediction(cfg):
    seed_everything(42)

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    val_loader = get_val_loader(tokenizer)
    model, model_taking_encoded_images = get_model(tokenizer)

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

                padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                coord = torch.cat([coord, padd], dim=0)
                coords.append(coord)
            batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]     

            for ip, pp in enumerate(batch_polygons):
                polys = []
                for p in pp:
                    p = torch.fliplr(p)
                    p = p[p[:, 0] != CFG.PAD_IDX]
                    p = p * (CFG.IMG_SIZE / CFG.INPUT_WIDTH)
                    polys.append(p)
                predictions.extend(generate_coco_ann(polys,idx[ip].item()))

            B, C, H, W = x.shape

            polygons_mask = np.zeros((B, 1, H, W))
            for b in range(len(batch_polygons)):
                for c in range(len(batch_polygons[b])):
                    poly = batch_polygons[b][c]
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    cnt = np.flip(np.int32(poly.cpu()), 1)
                    if len(cnt) > 0:
                        cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.)
            polygons_mask = torch.from_numpy(polygons_mask)

            batch_miou = mean_iou_metric(polygons_mask, y_mask)
            batch_macc = mean_acc_metric(polygons_mask, y_mask)

            cumulative_miou.append(batch_miou)
            cumulative_macc.append(batch_macc)

            pred_grid = make_grid(polygons_mask).permute(1, 2, 0)
            gt_grid = make_grid(y_mask).permute(1, 2, 0)
            plt.subplot(211), plt.imshow(pred_grid) ,plt.title("Predicted Polygons") ,plt.axis('off')
            plt.subplot(212), plt.imshow(gt_grid) ,plt.title("Ground Truth") ,plt.axis('off')
            
            img_outfile = os.path.join(cfg.output_dir, "images", f"predictions_{i_batch}.png")
            os.makedirs(os.path.dirname(img_outfile), exist_ok=True)
            plt.savefig(img_outfile)
            plt.close()

        print("Average model speed: ", np.mean(speed) / cfg.batch_size, " [s / image]")

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
    print("\nFinal Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Accessing nested values
    print(f"\nModel: {cfg.model.name}")
    # print(f"Layers: {cfg.model.architecture.num_layers}")
    # print(f"Pretrained on: {cfg.model.pretrained.dataset}")
    # print(f"Fine-tuning enabled: {cfg.model.pretrained.fine_tune.enabled}")
    run_prediction(cfg)

if __name__ == "__main__":
    main()
