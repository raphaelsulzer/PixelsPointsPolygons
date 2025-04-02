# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import time
import json
import cv2
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

from ..models.pix2poly.tokenizer import Tokenizer
from ..datasets import get_val_loader

from .predictor import Predictor

class Pix2PolyPredictor(Predictor):

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
    
    
    def predict_from_loader(self, model, tokenizer, loader):
        
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the dataset. Your coco evaluation will not be very useful.")
        
        model.eval()
        
        coco_predictions = []
        for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in self.progress_bar(loader):
            
            if self.cfg.use_images:
                x_image = x_image.to(self.cfg.device, non_blocking=True)
            if self.cfg.use_lidar:
                x_lidar = x_lidar.to(self.cfg.device, non_blocking=True)

            batch_preds, batch_confs, perm_preds = test_generate(
                model,x_image,x_lidar,tokenizer,
                max_len=self.cfg.model.tokenizer.generation_steps,top_k=0,top_p=1)
            
            
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
                file_names = get_tile_name_from_dataloader(loader.dataset.coco.imgs, image_ids)
                file_names = None
                plot_pix2poly(image_batch=x_image, tile_names=file_names, mask_batch=polygons_mask, polygon_batch=batch_polygons_processed)
                
        return coco_predictions
    

    def predict(self):
        
        self.logger.error("Predictor.predict() is probably not working. This will have to be implement for test.")

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
        
        
        ### TODO: get the model from the trainer, maybe inherit predictor from trainer.
        model = get_model(self.cfg,tokenizer=tokenizer)
        self.load_checkpoint(model)

        with torch.no_grad():
            t0 = time.time()
            coco_predictions = self.predict_from_loader(model,tokenizer,val_loader)

        prediction_outfile = os.path.join(self.cfg.output_dir, "predictions", f"{self.cfg.checkpoint}.json")
        with open(prediction_outfile, "w") as fp:
            fp.write(json.dumps(coco_predictions))

        self.logger.debug(f"Average prediction speed: {(time.time() - t0) / len(val_loader.dataset):.2f} [s / image]")
