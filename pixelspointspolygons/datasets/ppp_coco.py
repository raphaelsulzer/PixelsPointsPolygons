import os
import laspy
import cv2
import torch
import rasterio

import numpy as np
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset

from ..misc import make_logger, suppress_stdout


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class PPPDataset(Dataset):
    def __init__(self, cfg, split,
                 transform=None,
                 **kwargs):
        super().__init__()
        
        self.logger = make_logger(f'{split}Dataset', cfg.run_type.logging)
        
        self.cfg = cfg
        self.split = split

        self.dataset_dir = self.cfg.dataset.path
        if not os.path.isdir(self.dataset_dir):
            raise NotADirectoryError(f"Dataset directory {self.dataset_dir} does not exist")
        
        ## FFL currently still has a specific annotations file which includes the path to the .pt file with the frame field stored
        if self.cfg.experiment.model.name == "ffl":
            # self.ann_file = os.path.join(self.dataset_dir,f"annotations_ffl_{split}.json")
            self.ann_file = self.cfg.dataset.annotations[split].replace("annotations_", "annotations_ffl_")
            self.stats_filepath = self.cfg.dataset.ffl_stats[split]
            if not os.path.isfile(self.stats_filepath):
                # TODO: now there is no FFL training data for 224x224 tiles. Just include the FFL training data creation in ppp_dataset now
                raise FileExistsError(self.stats_filepath)
            self.stats = torch.load(self.stats_filepath)
        else:
            self.ann_file = self.cfg.dataset.annotations[split]
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError(self.ann_file)

        with suppress_stdout():
            self.coco = COCO(self.ann_file)
        images_id = self.coco.getImgIds()
        self.tile_ids = images_id.copy()
        self.num_samples = len(self.tile_ids)

        self.logger.info(f"Loaded {len(self.coco.anns.items())} annotations from {len(self.coco.imgs.items())} images from {self.ann_file}")

        self.use_lidar = cfg.use_lidar
        self.use_images = cfg.use_images
        self.transform = transform
        self.model_type = cfg.experiment.model.name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return self.num_samples
    
    def load_lidar_points(self, lidar_file_name, img_info):

        if os.path.isfile(lidar_file_name):
            
            las = laspy.read(lidar_file_name)

            points = np.vstack((las.x, las.y, las.z)).transpose()

            ### stop doing this scaling here
            points[:, :2] = (points[:, :2] - img_info['top_left']) / img_info.get('res_x', 0.25)
            points[:, 1] = img_info['height'] - points[:, 1]

            # scale z vals to [0,100]
            scaler = MinMaxScaler(feature_range=(0,self.cfg.experiment.encoder.in_voxel_size.z))
            points[:, -1] = scaler.fit_transform(points[:, -1].reshape(-1, 1)).squeeze()
            
            points = points.astype(np.float32)
            
            assert (points.min(axis=0) >= [-0.01,-0.01,-0.01]).all(), f"Points min {points.min(axis=0)} is not allowed."
            assert (points.max(axis=0) <= [img_info['width'],img_info['height'],self.cfg.experiment.encoder.in_voxel_size.z]).all(), f"Points max {points.min(axis=0)} is not allowed."

            points[:,0] = np.clip(points[:,0],0,img_info['width'])
            points[:,1] = np.clip(points[:,1],0,img_info['height'])
            
            # if not len(points) > 3:
            #     self.logger.warning(f"Lidar file {lidar_file_name} only has {len(points)} points.")

        else:
            raise FileExistsError(f"Lidar file {lidar_file_name} missing.")

        return points

    def get_image_file(self, coco_info):
        filename = os.path.join(self.dataset_dir, coco_info['image_path'])
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        return filename

    def get_lidar_file(self, coco_info):
        filename = os.path.join(self.dataset_dir, coco_info['lidar_path'])
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{filename}")
        return filename
    
    def apply_augmentations_to_lidar(self, augmentation_replay, lidar):
        
        if self.split != 'train':
            return torch.from_numpy(lidar)
    
        d4_transform = augmentation_replay["transforms"][0]
        
        assert d4_transform["__class_fullname__"] == "D4"
        

        if not d4_transform['applied']:
            return torch.from_numpy(lidar)
        
        # translate to center so all the transformations are easily applied around the center
        center = [self.cfg.experiment.encoder.in_width // 2, self.cfg.experiment.encoder.in_height // 2]
        lidar[:, :2] -= center
        
        group_element = d4_transform['params']['group_element']
        if group_element == 'e':
            # Identity, no change
            pass
        elif group_element == 'r90':
            lidar[:, [0, 1]] = lidar[:, [1, 0]]
            lidar[:, 1] = -lidar[:, 1]
        elif group_element == 'r180':
            lidar[:, 0] = -lidar[:, 0]
            lidar[:, 1] = -lidar[:, 1]
        elif group_element == 'r270':
            lidar[:, [0, 1]] = lidar[:, [1, 0]]
            lidar[:, 0] = -lidar[:, 0]
        elif group_element == 'v':
            lidar[:, 1] = -lidar[:, 1]
        elif group_element == 'hvt':
            lidar[:, [0, 1]] = lidar[:, [1, 0]]
            lidar[:, 0] = -lidar[:, 0]
            lidar[:, 1] = -lidar[:, 1]
        elif group_element == 'h':
            lidar[:, 0] = -lidar[:, 0]
        elif group_element == 't':
            lidar[:, [0, 1]] = lidar[:, [1, 0]]
        else:
            raise ValueError(f"Unknown group element {group_element}")
        
        lidar[:, :2] += center

        # self.logger.debug(f"Applied '{group_element}' transformation to lidar tile {id}.")
        
        return torch.from_numpy(lidar)
    
    
    def apply_augmentations_to_ffl_crossfield_angle(self, crossfield_angle_mask, augmentation_replay=None, group_element=None):
        
        if self.split == 'val':
            return crossfield_angle_mask
    
        if augmentation_replay is not None:
            d4_transform = augmentation_replay["transforms"][0]
            
            if d4_transform["__class_fullname__"] != "D4":
                # self.logger.warning(f"No D4 transform applied.")
                return crossfield_angle_mask
            
            if not d4_transform['applied']:
                return crossfield_angle_mask
        
        if group_element is None:
            group_element = d4_transform['params']['group_element']
        
        # self.logger.debug(f"Apply {group_element} augmentation")
        
        if group_element == 'e':
            # Identity, no change
            pass
        elif group_element == 'r90':
            crossfield_angle_mask = (crossfield_angle_mask + np.pi / 2) % np.pi
        elif group_element == 'r180':
            crossfield_angle_mask = (crossfield_angle_mask + np.pi) % np.pi
        elif group_element == 'r270':
            crossfield_angle_mask = (crossfield_angle_mask + 3 * np.pi / 2) % np.pi
        elif group_element == 'v':
            crossfield_angle_mask = (np.pi - crossfield_angle_mask) % np.pi
        elif group_element == 'hvt':
            crossfield_angle_mask = (3 * np.pi / 2 - crossfield_angle_mask) % np.pi
        elif group_element == 'h':
            crossfield_angle_mask = (-crossfield_angle_mask) % np.pi
        elif group_element == 't':
            crossfield_angle_mask = (np.pi / 2 - crossfield_angle_mask) % np.pi
        else:
            raise ValueError(f"Unknown group element {group_element}")

        return crossfield_angle_mask

    def __getitem__(self, idx):

        if self.model_type == 'hisup':
            return self.__getitem__hisup(idx)
        elif self.model_type == 'pix2poly':
            return self.__getitem__pix2poly(idx)
        elif self.model_type == 'ffl':
            return self.__getitem__ffl(idx)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented.")
        
    
    def __getitem__ffl(self, index):

        img_id = self.tile_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        # load image
        if self.use_images:
            img_file = self.get_image_file(img_info)
            # image = np.array(Image.open(img_file).convert("RGB"))
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])  # shape (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
        else:
            # make dummy image for albumentations to work
            image = np.zeros((img_info['width'], 
                            img_info['height'], 1), dtype=np.uint8)

        # load lidar
        if self.use_lidar:
            filename = self.get_lidar_file(img_info)
            lidar = self.load_lidar_points(filename, img_info)
        else:
            lidar = None
            
        ### TODO: integrate the FFL GT into the standard COCO annotations file in ppp_dataset code
        # get ffl_pt file
        ffl_pt_file = os.path.join(self.dataset_dir,img_info["ffl_pt_path"])
        if not os.path.isfile(ffl_pt_file):
            raise FileExistsError(ffl_pt_file)
        ffl_data = torch.load(ffl_pt_file,weights_only=False)
        
        ffl_data["image_id"] = torch.IntTensor([img_id])
        
        if self.transform is not None: 
            
            masks = []
            masks.append(ffl_data["gt_polygons_image"][:,:,0])  # interior
            masks.append(ffl_data["gt_polygons_image"][:,:,1])  # edges
            masks.append(ffl_data["gt_polygons_image"][:,:,2])  # vertices
            masks.append(ffl_data["distances"])
            masks.append(ffl_data["sizes"])
            masks.append(ffl_data["gt_crossfield_angle"])
            
            augmentations = self.transform(image=image,masks=masks)
            
            if self.use_lidar:
                ffl_data["lidar"] = self.apply_augmentations_to_lidar(augmentations["replay"], lidar)
            
            ffl_data["image"] = augmentations['image']
            
            gt_polygon_image = []
            for i in range(3):
                gt_polygon_image.append(augmentations['masks'][i])
            ffl_data["gt_polygons_image"] = torch.stack(gt_polygon_image, axis=-1).permute(2,0,1)
            ffl_data["gt_polygons_image"] = ffl_data["gt_polygons_image"]/255
            ffl_data["gt_polygons_image"] = torch.clamp(ffl_data["gt_polygons_image"],0,1)
            ffl_data["gt_polygons_image"] = ffl_data["gt_polygons_image"].to(torch.float32)
            
            # used for optional seg_loss weighting
            ffl_data["distances"] = augmentations['masks'][3][None, ...]
            
            # used for optional seg_loss weighting
            ffl_data["sizes"] = augmentations['masks'][4][None,...]
            
            # first bring the values back to [0,180] from 8bit
            ffl_data["gt_crossfield_angle"] = augmentations['masks'][5] * np.pi / 255.0
            
            # for some reason the normals instead of tangents are stored in the .pt file
            ffl_data["gt_crossfield_angle"] = (ffl_data["gt_crossfield_angle"] + np.pi / 2) % np.pi
            
            # rotate the angles inside the crossfield            
            ffl_data["gt_crossfield_angle"] = self.apply_augmentations_to_ffl_crossfield_angle(ffl_data["gt_crossfield_angle"],
                                                                                               augmentation_replay=augmentations["replay"])
            ffl_data["gt_crossfield_angle"] = ffl_data["gt_crossfield_angle"][None,...]
            
        ffl_data["class_freq"] = torch.from_numpy(self.stats["class_freq"])
        
        
        return ffl_data
        
    
    def __getitem__pix2poly(self, index):
        
        if not hasattr(self,"tokenizer"):
            raise ValueError("Tokenizer not set. Please pass a tokenizer to the dataset class when using Pix2Poly.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please pass a tokenizer to the dataset class when using Pix2Poly.")

        img_id = self.tile_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(ann_ids)  # annotations of all instances in an image.

        # load image
        if self.use_images:
            img_file = self.get_image_file(img_info)
            # image = np.array(Image.open(img_file).convert("RGB"))
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])  # shape (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
        else:
            # make dummy image for albumentations to work
            image = np.zeros((img_info['width'], 
                            img_info['height'], 1), dtype=np.uint8)

        # load lidar
        if self.use_lidar:
            img_file = self.get_lidar_file(img_info)
            lidar = self.load_lidar_points(img_file, img_info)
        else:
            lidar = None

        mask = np.zeros((img_info['width'], img_info['height']))
        corner_coords = []
        corner_mask = np.zeros((img_info['width'], img_info['height']), dtype=np.float32)
        perm_matrix = np.zeros((self.cfg.experiment.model.tokenizer.n_vertices, self.cfg.experiment.model.tokenizer.n_vertices), dtype=np.float32)
        for ins in annotations:
            segmentations = ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                segm[:, 0] = np.clip(segm[:, 0], 0, img_info['width'] - 1)
                segm[:, 1] = np.clip(segm[:, 1], 0, img_info['height'] - 1)
                points = segm[:-1]
                corner_coords.extend(points.tolist())
                mask += self.coco.annToMask(ins)
        mask = mask / 255. if mask.max() == 255 else mask
        mask = np.clip(mask, 0, 1)

        corner_coords = np.flip(np.round(corner_coords, 0), axis=-1).astype(np.int32)

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
                    if v_count + i > self.cfg.experiment.model.tokenizer.n_vertices - 1 or v_count + j > self.cfg.experiment.model.tokenizer.n_vertices - 1:
                        break
                    perm_matrix[v_count + i, v_count + j] = 1.
                v_count += len(points)

        for i in range(v_count, self.cfg.experiment.model.tokenizer.n_vertices):
            perm_matrix[i, i] = 1.

        # Workaround for open contours:
        for i in range(self.cfg.experiment.model.tokenizer.n_vertices):
            row = perm_matrix[i, :]
            col = perm_matrix[:, i]
            if np.sum(row) == 0 or np.sum(col) == 0:
                perm_matrix[i, i] = 1.
        perm_matrix = torch.from_numpy(perm_matrix)
        ############# END: Generate gt permutation matrix. #############

        masks = [mask, corner_mask]

        if len(corner_coords) > self.cfg.experiment.model.tokenizer.n_vertices:
            corner_coords = corner_coords[:self.cfg.experiment.model.tokenizer.n_vertices]

        if self.transform is not None: 
            
            augmentations = self.transform(image=image, masks=masks, keypoints=corner_coords.tolist())
            
            if self.use_lidar:
                lidar = self.apply_augmentations_to_lidar(augmentations["replay"], lidar)
            
            image = augmentations['image']
            mask = augmentations['masks'][0]
            corner_mask = augmentations['masks'][1]
            corner_coords = np.array(augmentations['keypoints'])

        coords_seqs, rand_idxs = self.tokenizer(corner_coords, shuffle=self.cfg.experiment.model.tokenizer.shuffle_tokens)
        coords_seqs = torch.LongTensor(coords_seqs)
        if self.cfg.experiment.model.tokenizer.shuffle_tokens:
            perm_matrix = self.shuffle_perm_matrix_by_indices(perm_matrix, rand_idxs)

        return image, lidar, mask[None, ...], corner_mask[None, ...], coords_seqs, perm_matrix, torch.tensor([img_info['id']])


    def __getitem__hisup(self, index):

        img_id = self.tile_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(ann_ids)  # annotations of all instances in an image.

        # load image
        if self.use_images:
            img_file = self.get_image_file(img_info)
            # image = np.array(Image.open(img_file).convert("RGB"))            
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])  # shape (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, 3)

        else:
            # make dummy image for albumentations to work
            image = np.zeros((img_info['width'], 
                            img_info['height'], 1), dtype=np.uint8)

        # load lidar
        if self.use_lidar:
            img_file = self.get_lidar_file(img_info)
            lidar = self.load_lidar_points(img_file, img_info)
        else:
            lidar = None
        

        corner_coords = []
        corner_poly_ids = [0]
        mask = np.zeros([img_info['width'], img_info['height']])
        for i,annotation_per_image in enumerate(annotations):
            mask += self.coco.annToMask(annotation_per_image)
            segmentations = annotation_per_image['segmentation']
            if len(segmentations) > 1:
                raise ValueError("Only one segmentation per instance is supported. This is a multipolygon.")
            points = segmentations[0]
            points = np.array(points).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, img_info['width'] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, img_info['height'] - 1)
            points = points[:-1]
            corner_poly_ids.append(len(points)+len(corner_coords))
            corner_coords.extend(points.tolist())
        mask = mask / 255. if mask.max() == 255 else mask
        mask = np.clip(mask, 0, 1)
        
        if self.transform is not None: 
            
            corner_coords = np.flip(corner_coords,axis=-1)
            augmentations = self.transform(image=image, masks=[mask], keypoints=corner_coords)
                        
            if self.use_lidar:
                lidar = self.apply_augmentations_to_lidar(augmentations["replay"], lidar)
            
            image = augmentations['image']
            corner_coords = np.flip(augmentations['keypoints'],axis=-1)

        annotations = self.make_hisup_annotations(corner_coords, corner_poly_ids, img_info['height'], img_info['width'])
        annotations["mask"] = augmentations['masks'][0]
        
        if self.cfg.experiment.model.decoder.in_feature_width != img_info['width'] or self.cfg.experiment.model.decoder.in_feature_height != img_info['height']:
            self.resize_hisup_annotations(annotations)
        else:
            annotations['mask_ori'] = annotations['mask'].clone()
        
        for k, v in annotations.items():
            if isinstance(v, np.ndarray):
                annotations[k] = torch.from_numpy(v)

        return image, lidar, annotations, torch.tensor([img_info['id']])


    def resize_hisup_annotations(self,ann):
        
        sx = self.cfg.experiment.model.decoder.in_feature_width / ann['width']
        sy = self.cfg.experiment.model.decoder.in_feature_height / ann['height']
        ann['junc_ori'] = ann['junctions'].copy()
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.cfg.experiment.model.decoder.in_feature_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.cfg.experiment.model.decoder.in_feature_height - 1e-4)
        ann['width'] = self.cfg.experiment.model.decoder.in_feature_width
        ann['height'] = self.cfg.experiment.model.decoder.in_feature_height
        ann['mask_ori'] = ann['mask'].clone()
        ann['mask'] = cv2.resize(np.array(ann['mask']).astype(np.uint8), (int(self.cfg.experiment.model.decoder.in_feature_width), int(self.cfg.experiment.model.decoder.in_feature_height)))
    
    
    def make_hisup_annotations(self, corner_coords, corner_poly_ids, height, width):

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
        for i,_ in enumerate(corner_poly_ids[:-1]):
            
            juncs, tags = [], []

            points = corner_coords[corner_poly_ids[i] : corner_poly_ids[i+1]]
            
            junc_tags = np.ones(points.shape[0])
            
            poly = Polygon(points)
            if poly.area > 0:
                convex_point = np.array(poly.convex_hull.exterior.coords)
                convex_point = convex_point[:-1]
                convex_index = [(p == convex_point).all(1).any() for p in points]
                
                # self.plot_hisup_poly(points, convex_index)
                
                juncs.extend(points.tolist())
                junc_tags[convex_index] = 2  # convex point label
                tags.extend(junc_tags.tolist())
                ann['bbox'].append(list(poly.bounds))


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

        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.longlong],
                           ['juncs_tag', np.longlong],
                           ['juncs_index', np.longlong],
                           ['bbox', np.float32],
                           ):
            ann[key] = np.array(ann[key], dtype=_type)

        if len(ann['junctions']) == 0:
            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0, 0, 0, 0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])

        return ann
    
    

class TestDataset(PPPDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'test',**kwargs)

class ValDataset(PPPDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'val',**kwargs)

class TrainDataset(PPPDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'train',**kwargs)