# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import logging
import time
import json
import cv2
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

from tqdm import tqdm

from .misc import *
from .models.tokenizer import Tokenizer
from .datasets import get_val_loader
from .models import get_model

# import warnings
# warnings.filterwarnings("error", message="Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.")

class Predictor:
    def __init__(self, cfg, verbosity=logging.INFO):
        self.cfg = cfg
        # self.logger = make_logger("Prediction",level=verbosity,filepath=os.path.join(cfg.output_dir, 'predict.log'))
        self.logger = make_logger("Prediction",level=verbosity)
        self.logger.info("Create output directory {cfg.output_dir}...")
        os.makedirs(cfg.output_dir, exist_ok=True)

    def get_pixel_mask_from_prediction(self, x, batch_polygons):
        B, C, H, W = x.shape

        polygons_mask = np.zeros((B, 1, H, W))
        for b in range(len(batch_polygons)):
            for c in range(len(batch_polygons[b])):
                poly = batch_polygons[b][c]
                poly = poly[poly[:, 0] != self.cfg.model.tokenizer.pad_idx]
                cnt = np.flip(np.int32(poly.cpu()), 1)
                if len(cnt) > 0:
                    cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.)
        return torch.from_numpy(polygons_mask)
    
    def load_checkpoint(self, model):
        
        if self.cfg.checkpoint_file is not None:
            checkpoint_file = self.cfg.checkpoint_file
            self.cfg.checkpoint = os.path.basename(checkpoint_file).split(".")[0]+"_overwrite"
        else:
            checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"{self.cfg.checkpoint}.pth")
        if not os.path.isfile(checkpoint_file):
            raise FileExistsError(f"Checkpoint file {checkpoint_file} not found.")
        self.logger.info(f"Loading model from checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.cfg.device)
        single_gpu_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        # single_gpu_state_dict = checkpoint["state_dict"]
        model.load_state_dict(single_gpu_state_dict)
        epoch = checkpoint['epochs_run']
        self.logger.info(f"Model loaded from epoch: {epoch}")
        

    def predict(self):

        tokenizer = Tokenizer(
            num_classes=1,
            num_bins=self.cfg.model.tokenizer.num_bins,
            width=self.cfg.model.encoder.input_width,
            height=self.cfg.model.encoder.input_height,
            max_len=self.cfg.model.tokenizer.max_len
        )
        
        compute_dynamic_cfg_vars(self.cfg,tokenizer)

        val_loader = get_val_loader(self.cfg,tokenizer)
        # val_loader = get_train_loader(self.cfg,tokenizer)
        model = get_model(self.cfg,tokenizer=tokenizer)
        self.load_checkpoint(model)

        with torch.no_grad():
            speed = []
            coco_predictions = []
            for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in tqdm(val_loader):
                t0 = time.time()
                
                if self.cfg.use_images:
                    x_image = x_image.to(self.cfg.device, non_blocking=True)
                if self.cfg.use_lidar:
                    x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)

                batch_preds, batch_confs, perm_preds = test_generate(
                    model,x_image,x_lidar,tokenizer,
                    max_len=self.cfg.model.tokenizer.generation_steps,top_k=0,top_p=1)
                
                speed.append(time.time() - t0)
                vertex_coords, _ = postprocess(batch_preds, batch_confs, tokenizer)

                coords = []
                for i in range(len(vertex_coords)):
                    if vertex_coords[i] is not None:
                        coord = torch.from_numpy(vertex_coords[i])
                    else:
                        coord = torch.tensor([])
                    padd = torch.ones((self.cfg.model.tokenizer.n_vertices - len(coord), 2)).fill_(self.cfg.model.tokenizer.pad_idx)
                    coord = torch.cat([coord, padd], dim=0)
                    coords.append(coord)
                    
                batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]     

                batch_polygons_processed = []
                for i, pp in enumerate(batch_polygons):
                    polys = []
                    for p in pp:
                        p = torch.fliplr(p)
                        p = p[p[:, 0] != self.cfg.model.tokenizer.pad_idx]
                        if len(p) > 0:
                            polys.append(p)
                    batch_polygons_processed.append(polys)
                    coco_predictions.extend(generate_coco_ann(polys,image_ids[i].item()))


                if self.cfg.debug_vis:
                    polygons_mask = self.get_pixel_mask_from_prediction(x_image,batch_polygons)
                    file_names = get_image_file_name_from_dataloader(val_loader.dataset.coco.imgs, image_ids)
                    plot_pix2poly(image_batch=x_image, image_names=file_names, mask_batch=polygons_mask, polygon_batch=batch_polygons_processed)

            self.logger.debug("Average model speed: ", np.mean(speed) / self.cfg.model.batch_size, " [s / image]")

        prediction_outfile = os.path.join(self.cfg.output_dir, f"predictions_{self.cfg.checkpoint}.json")
        with open(prediction_outfile, "w") as fp:
            fp.write(json.dumps(coco_predictions))
