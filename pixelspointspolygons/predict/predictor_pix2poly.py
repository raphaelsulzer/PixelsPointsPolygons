# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import time
import json
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

from scipy.optimize import linear_sum_assignment
from transformers.generation.utils import top_k_top_p_filtering

from ..models.pix2poly import Tokenizer, Pix2PolyModel
from ..misc.coco_conversions import generate_coco_ann
from ..datasets import get_val_loader
from ..datasets import get_train_loader, get_val_loader, get_test_loader

from .predictor import Predictor

class Pix2PolyPredictor(Predictor):
    
    def setup_tokenizer(self):
        self.tokenizer = Tokenizer(num_classes=1,
            num_bins=self.cfg.experiment.model.tokenizer.num_bins,
            width=self.cfg.experiment.encoder.in_width,
            height=self.cfg.experiment.encoder.in_height,
            max_len=self.cfg.experiment.model.tokenizer.max_len
        )
        self.cfg.experiment.model.tokenizer.pad_idx = self.tokenizer.PAD_code
        self.cfg.experiment.model.tokenizer.max_len = self.cfg.experiment.model.tokenizer.n_vertices*2+2
        self.cfg.experiment.model.tokenizer.generation_steps = self.cfg.experiment.model.tokenizer.n_vertices*2+1
        
    def setup_model_and_load_checkpoint(self):
        
        self.setup_tokenizer()
        self.model = Pix2PolyModel(self.cfg,self.tokenizer.vocab_size,local_rank=self.local_rank)
        self.model.eval()
        self.model.to(self.cfg.host.device)
        self.load_checkpoint()
    
    
    def predict_dataset(self, split="val"):
        
        self.setup_model_and_load_checkpoint()
        
        if split == "train":
            self.loader = get_train_loader(self.cfg,tokenizer=self.tokenizer,logger=self.logger)
        elif split == "val":
            self.loader = get_val_loader(self.cfg,tokenizer=self.tokenizer,logger=self.logger)
        elif split == "test":
            self.loader = get_test_loader(self.cfg,tokenizer=self.tokenizer,logger=self.logger)
        else:   
            raise ValueError(f"Unknown split {split}.")
        
        self.logger.info(f"Predicting on {len(self.loader)} batches...")
        
        with torch.no_grad():
            t0 = time.time()
            coco_predictions = self.predict_from_loader(self.model,self.tokenizer,self.loader)

        self.logger.info(f"Average prediction speed: {(time.time() - t0) / len(self.loader.dataset):.2f} [s / image]")
        time_dict = {}
        time_dict["prediction_time"] = (time.time() - t0) / len(self.loader.dataset)
        
        if self.local_rank == 0:
            prediction_outfile = self.cfg.evaluation.pred_file
            self.logger.info(f"Saving predictions to {prediction_outfile}")
            os.makedirs(os.path.dirname(prediction_outfile), exist_ok=True)
            with open(prediction_outfile, "w") as fp:
                fp.write(json.dumps(coco_predictions))
        

        return time_dict

    
    def predict_from_loader(self, model, tokenizer, loader):
                
        if isinstance(loader.dataset, torch.utils.data.Subset):
            self.logger.warning("You are predicting only a subset of the dataset. Your coco evaluation will not be very useful.")
        
        model.eval()
        
        coco_predictions = []
        for x_image, x_lidar, y_mask, y_corner_mask, y_sequence, y_perm, image_ids in self.progress_bar(loader):
            
            if self.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self.device, non_blocking=True)
            
            batch_polygons = self.batch_to_polygons(x_image, x_lidar, model, tokenizer)
            
            for i, polys in enumerate(batch_polygons):
                coco_predictions.extend(generate_coco_ann(polys,image_ids[i].item()))
                
        return coco_predictions


    def predict_file(self,img_infile=None,lidar_infile=None,outfile=None):
                
        image, image_pillow = self.load_image_from_file(img_infile)
        lidar = self.load_lidar_from_file(lidar_infile)

        self.setup_model_and_load_checkpoint()
        
        with torch.no_grad():
            
            if self.cfg.experiment.encoder.use_images:
                image = image.to(self.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                lidar = lidar.to(self.device, non_blocking=True)
            
            batch_polygons = self.batch_to_polygons(image, lidar, self.model, self.tokenizer)
            self.plot_prediction(batch_polygons[0], image=image, image_np=image_pillow, lidar=lidar, outfile=outfile)

            
            
    def batch_to_polygons(self, x_images, x_lidar, model, tokenizer):
        """Takes one batch with samples of images and/or lidar data and returns the polygons for each sample of the batch."""
        
        ### need this so I do not have to pass the model and tokenizer to the several polygonization functions
        self.tokenizer = tokenizer
        self.model = model
        
        batch_preds, batch_confs, perm_preds = self.test_generate(x_images,x_lidar)
        
        vertex_coords, _ = self.postprocess(batch_preds, batch_confs)

        coords = []
        for i in range(len(vertex_coords)):
            if vertex_coords[i] is not None:
                coord = torch.from_numpy(vertex_coords[i])
            else:
                coord = torch.tensor([])
            padd = torch.ones((self.cfg.experiment.model.tokenizer.n_vertices - len(coord), 2)).fill_(self.cfg.experiment.model.tokenizer.pad_idx)
            coord = torch.cat([coord, padd], dim=0)
            coords.append(coord)
            
        batch_polygons = self.permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]     

        batch_polygons_processed = []
        for i, pp in enumerate(batch_polygons):
            polys = []
            for p in pp:
                p = torch.fliplr(p)
                p = p[p[:, 0] != self.cfg.experiment.model.tokenizer.pad_idx]
                if len(p) > 0:
                    polys.append(p)
            
            batch_polygons_processed.append(polys)
            
        return batch_polygons_processed
        


    def test_generate(self, x_images, x_lidar, top_k=0, top_p=1):
        
        batch_size = x_images.size(0) if x_images is not None else x_lidar.size(0)
        
        batch_preds = torch.ones((batch_size, 1), device=self.device).fill_(self.tokenizer.BOS_code).long()

        confs = []

        if top_k != 0 or top_p != 1:
            sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
        else:
            sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

        with torch.no_grad():
            ## a bit ugly :/
            if self.cfg.host.multi_gpu:
                
                if self.cfg.experiment.encoder.use_images and not self.cfg.experiment.encoder.use_lidar:
                    features = self.model.module.encoder(x_images)
                elif not self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
                    features = self.model.module.encoder(x_lidar)
                elif self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
                    features = self.model.module.encoder(x_images, x_lidar)
                else:
                    raise ValueError("At least one of use_images or use_lidar must be True")
                
            else:
                
                if self.cfg.experiment.encoder.use_images and not self.cfg.experiment.encoder.use_lidar:
                    features = self.model.encoder(x_images)
                elif not self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
                    features = self.model.encoder(x_lidar)
                elif self.cfg.experiment.encoder.use_images and self.cfg.experiment.encoder.use_lidar:
                    features = self.model.encoder(x_images, x_lidar)
                else:
                    raise ValueError("At least one of use_images or use_lidar must be True")
                
                
            for i in range(self.cfg.experiment.model.tokenizer.generation_steps):
                if self.cfg.host.multi_gpu:
                    preds, feats = self.model.module.predict(features, batch_preds)
                else:
                    preds, feats = self.model.predict(features, batch_preds)
                    
                preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)  # if top_k and top_p are set to default, this line does nothing.
                if i % 2 == 0:
                    confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                    confs.append(confs_)
                preds = sample(preds)
                batch_preds = torch.cat([batch_preds, preds], dim=1)

            if self.cfg.host.multi_gpu:
                perm_preds = self.model.module.scorenet1(feats) + torch.transpose(self.model.module.scorenet2(feats), 1, 2)
            else:
                perm_preds = self.model.scorenet1(feats) + torch.transpose(self.model.scorenet2(feats), 1, 2)

            perm_preds = self.scores_to_permutations(perm_preds)

        return batch_preds.cpu(), confs, perm_preds
    
    
    
    def permutations_to_polygons(self, perm, graph, out='torch'):
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






    def postprocess(self, batch_preds, batch_confs):
        EOS_idxs = (batch_preds == self.tokenizer.EOS_code).float().argmax(dim=-1)
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
            coords = self.tokenizer.decode(batch_preds[i, :EOS_idx+1])
            confs = [round(batch_confs[j][i].item(), 3) for j in range(len(coords))]

            all_coords.append(coords)
            all_confs.append(confs)

        return all_coords, all_confs

    def scores_to_permutations(self, scores):
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

