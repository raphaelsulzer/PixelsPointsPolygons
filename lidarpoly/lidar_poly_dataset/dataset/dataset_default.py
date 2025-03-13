import cv2
import random
import numpy as np
import os
import laspy
import logging
from PIL import Image
import torch

from skimage import io
from sklearn.preprocessing import MinMaxScaler
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lidar_poly_dataset.misc import make_logger

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


class DefaultDataset(Dataset):
    def __init__(self, cfg, split,
                 transform=None,
                 **kwargs):
        super().__init__()
        
        self.logger = make_logger(f'{split}Dataset', cfg.run_type.logging)
        
        self.cfg = cfg
        self.split = split

        self.dataset_dir = self.cfg.dataset.path
        if not os.path.isdir(self.dataset_dir):
            raise NotADirectoryError(self.dataset_dir)
        
        self.ann_file = os.path.join(self.dataset_dir,f"annotations_{split}.json")
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError(self.ann_file)
        
        
        self.coco = COCO(self.ann_file)
        images_id = self.coco.getImgIds()
        self.tile_ids = images_id.copy()
        self.num_samples = len(self.tile_ids)

        self.logger.debug(f"Loaded {len(self.coco.anns.items())} annotations from {self.ann_file}")

        self.use_lidar = cfg.use_lidar
        self.use_images = cfg.use_images
        self.transform = transform
        self.model_type = cfg.model.name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return self.num_samples
    
    def load_lidar_points(self, lidar_file_name, img_info, z_scale=(0, 100)):

        if os.path.isfile(lidar_file_name):
            
            las = laspy.read(lidar_file_name)

            points = np.vstack((las.x, las.y, las.z)).transpose()

            points[:, :2] = (points[:, :2] - img_info['top_left']) / img_info.get('res_x', 0.25)
            points[:, 1] = img_info['height'] - points[:, 1]

            # scale z vals to [0,100]
            scaler = MinMaxScaler(feature_range=z_scale)  # Change range as needed
            points[:, -1] = scaler.fit_transform(points[:, -1].reshape(-1, 1)).squeeze()
            
            points = points.astype(np.float32)

        else:
            raise FileExistsError(f"Lidar file {lidar_file_name} missing.")
            # self.logger.debug(f'Lidar file {lidar_file_name} missing. Generating random point cloud.')

            # n = 5
            # # make a random point cloud
            # x = np.random.uniform(0, img_info['width'], n)
            # y = np.random.uniform(0, img_info['height'], n)
            # z = np.random.uniform(z_scale[0], z_scale[1], n)
            # points = np.vstack((x, y, z)).T

        return points

    def get_image_file(self, coco_info):
        filename = os.path.join(self.dataset_dir, coco_info['image_path'])
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        return filename

    def get_lidar_file(self, coco_info):
        filename = os.path.join(self.dataset_dir, coco_info['lidar_path'])
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        return filename

    def __getitem__(self, idx):

        if self.model_type == 'hisup':
            pass
        elif self.model_type == 'pix2poly':
            return self.__getitem__pix2poly(idx)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented.")

    def __getitem__pix2poly(self, index):

        img_id = self.tile_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(ann_ids)  # annotations of all instances in an image.

        # load image
        if self.use_images:
            filename = self.get_image_file(img_info)
            image = np.array(Image.open(filename).convert("RGB"))
        else:
            image = None

        # load lidar
        if self.use_lidar:
            filename = self.get_lidar_file(img_info)
            lidar = self.load_lidar_points(filename, img_info)
        else:
            lidar = None

        mask = np.zeros((img_info['width'], img_info['height']))
        corner_coords = []
        corner_mask = np.zeros((img_info['width'], img_info['height']), dtype=np.float32)
        perm_matrix = np.zeros((self.cfg.model.tokenizer.n_vertices, self.cfg.model.tokenizer.n_vertices), dtype=np.float32)
        point_ids = []
        point_id = 0
        for ins in annotations:
            segmentations = ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                segm[:, 0] = np.clip(segm[:, 0], 0, img_info['width'] - 1)
                segm[:, 1] = np.clip(segm[:, 1], 0, img_info['height'] - 1)
                points = segm[:-1]
                corner_coords.extend(points.tolist())
                mask += self.coco.annToMask(ins)
                point_ids.extend([point_id] * len(points))
                point_id += 1
        mask = mask / 255. if mask.max() == 255 else mask
        mask = np.clip(mask, 0, 1)

        ### commented the line below, because I think it makes the polygon vertices axis order incoorect
        corner_coords = np.flip(np.round(corner_coords, 0), axis=-1).astype(np.int32)
        # corner_coords = np.round(corner_coords, 0).astype(np.int32)

        if len(corner_coords) > 0.:
            corner_mask[corner_coords[:, 0], corner_coords[:, 1]] = 1.

        ############# START: Generate gt permutation matrix. #############
        v_count = 0
        for ins in annotations:
            segmentations = ins['segmentation']
            for idx, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                points = segm[:-1]
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    if v_count + i > self.cfg.model.tokenizer.n_vertices - 1 or v_count + j > self.cfg.model.tokenizer.n_vertices - 1:
                        break
                    perm_matrix[v_count + i, v_count + j] = 1.
                v_count += len(points)

        for i in range(v_count, self.cfg.model.tokenizer.n_vertices):
            perm_matrix[i, i] = 1.

        # Workaround for open contours:
        for i in range(self.cfg.model.tokenizer.n_vertices):
            row = perm_matrix[i, :]
            col = perm_matrix[:, i]
            if np.sum(row) == 0 or np.sum(col) == 0:
                perm_matrix[i, i] = 1.
        perm_matrix = torch.from_numpy(perm_matrix)
        ############# END: Generate gt permutation matrix. #############

        masks = [mask, corner_mask]

        if len(corner_coords) > self.cfg.model.tokenizer.n_vertices:
            corner_coords = corner_coords[:self.cfg.model.tokenizer.n_vertices]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=masks, keypoints=corner_coords.tolist())
            image = augmentations['image']
            mask = augmentations['masks'][0]
            corner_mask = augmentations['masks'][1]
            corner_coords = np.array(augmentations['keypoints'])

            if self.cfg.use_lidar:
                lidar = torch.from_numpy(lidar)
            
            # if len(corner_coords) > 0:
            #     # self.debug_vis(corner_coords, point_ids, augmented_image)
            #     augmented_image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            #     self.debug_vis(np.flip(corner_coords,axis=-1), point_ids, augmented_image)
            #     self.debug_vis_mask(mask)
            #     self.debug_vis_mask(corner_mask)
            #     a=5

        if self.tokenizer is not None:
            coords_seqs, rand_idxs = self.tokenizer(corner_coords, shuffle=self.cfg.model.tokenizer.shuffle_tokens)
            coords_seqs = torch.LongTensor(coords_seqs)
            if self.cfg.model.tokenizer.shuffle_tokens:
                perm_matrix = self.shuffle_perm_matrix_by_indices(perm_matrix, rand_idxs)
        else:
            coords_seqs = corner_coords

        return image, lidar, mask[None, ...], corner_mask[None, ...], coords_seqs, perm_matrix, torch.tensor([img_info['id']])




    def make_hisup_annotations(self, ann_coco, height, width):

        ann = {
            'junctions': [],
            'juncs_index': [],
            'juncs_tag': [],
            'edges_positive': [],
            'bbox': [],
            'width': width,
            'height': height,
        }

        pid = 0
        instance_id = 0
        seg_mask = np.zeros([width, height])
        for ann_per_ins in ann_coco:
            juncs, tags = [], []
            segmentations = ann_per_ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)  # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
                points = segm[:-1]
                junc_tags = np.ones(points.shape[0])
                if i == 0:  # outline
                    poly = Polygon(points)
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2  # convex point label
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)], -1, color=0, thickness=-1)

            idxs = np.arange(len(juncs))
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1, 0) + pid

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        seg_mask = np.clip(seg_mask, 0, 1)
        ann['mask'] = seg_mask

        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.longlong],
                           ['juncs_tag', np.longlong],
                           ['juncs_index', np.longlong],
                           ['bbox', np.float32],
                           ):
            ann[key] = np.array(ann[key], dtype=_type)

        if len(ann['junctions']) == 0:
            ann['mask'] = np.zeros((height, width), dtype=np.float64)
            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0, 0, 0, 0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])

        return ann

