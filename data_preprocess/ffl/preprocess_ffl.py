import os
import pathlib
import warnings
import hydra
import json

import skimage.io
from multiprocessing import Pool
from functools import partial
import _pickle

import numpy as np
from pycocotools.coco import COCO
import shapely.geometry

import torch.utils
import torch.utils.data
from tqdm import tqdm

import torch

from omegaconf import OmegaConf
from copy import deepcopy

# from lydorn_utils import print_utils
# from lydorn_utils import python_utils

from torch_lydorn.torch.utils.data import Dataset as LydornDataset, __repr__

from torch_lydorn.torchvision.datasets import utils


import data_transforms

class DatasetWithPreprocessing(torch.utils.data.Dataset):
    def __init__(self, root, pre_transform, fold="train", pool_size=1):
        super().__init__()
        assert fold in ["train", "val", "test_images"], "Input fold={} should be in [\"train\", \"val\", \"test_images\"]".format(fold)

        self.root = root
        self.fold = fold
        os.makedirs(self.processed_dir, exist_ok=True)

        self.pool_size = pool_size

        self.coco = None
        self.image_id_list = self.load_image_ids()
        self.stats_filepath = os.path.join(self.processed_dir, "stats.pt")
        self.stats = None
        if os.path.exists(self.stats_filepath):
            self.stats = torch.load(self.stats_filepath)
        self.processed_flag_filepath = os.path.join(self.processed_dir, "processed-flag")

        self.pre_transform = pre_transform
        # super(DatasetPreprocessor, self).__init__(root, None, pre_transform)

    def load_image_ids(self):

        coco = self.get_coco()
        image_id_list = coco.getImgIds(catIds=coco.getCatIds())

        return image_id_list

    def get_coco(self):
        if self.coco is None:
            
            annotation_filename = f"annotations_{self.fold}_processed.json"
            annotations_filepath = os.path.join(self.root, annotation_filename)
            if os.path.isfile(annotations_filepath):
                print("There is already a processed annotation file. Using this one.")
            else:
                annotation_filename = f"annotations_{self.fold}.json"
                annotations_filepath = os.path.join(self.root, annotation_filename)
            if not os.path.isfile(annotations_filepath):
                raise FileNotFoundError(f"Annotation file {annotations_filepath} not found in {self.root}.")
            
            self.coco = COCO(annotations_filepath)
        return self.coco

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.fold)

    @property
    def processed_file_names(self):
        l = []
        for image_id in self.image_id_list:
            l.append(os.path.join("data_{:012d}.pt".format(image_id)))
        return l

    def __len__(self):
        return len(self.image_id_list)

    def _download(self):
        pass

    def download(self):
        pass

    def _process(self):
        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            warnings.warn(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            warnings.warn(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if os.path.exists(self.processed_flag_filepath):
            print("Dataset already processed. Skipping pre-processing...")
            return

        print('Pre-Processing...')

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        
        print('Done!')

    def process(self):
        

        image_info_list = []
        image_info_with_pt_file_list = []
        coco = self.get_coco()
        for image_id in self.image_id_list:
            image_info = coco.loadImgs(image_id)[0]
            image_info_with_pt_file = deepcopy(image_info)
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotation_list = coco.loadAnns(annotation_ids)
            image_info["annotation_list"] = annotation_list
            image_info["absolute_img_filepath"] = os.path.join(self.root, image_info["file_name"])
            
            image_info["name"] = os.path.basename(image_info["file_name"]).replace(".tif", ".pt")
            image_info["pt_outfile"] = os.path.join(self.processed_dir, image_info["name"])
            
            image_info_list.append(image_info)

            image_info_with_pt_file["ffl_pt_path"] = f"processed/{self.fold}/{image_info['name']}"
            image_info_with_pt_file_list.append(image_info_with_pt_file)
            
        
        if self.pool_size > 0:
            partial_preprocess_one = partial(preprocess_one, pre_transform=self.pre_transform)
            with Pool(self.pool_size) as p:
                sample_stats_list = list(tqdm(p.imap(partial_preprocess_one, image_info_list), total=len(image_info_list)))
        else:
            sample_stats_list = []
            for image_info in tqdm(image_info_list):
                sample_stats_list.append(preprocess_one(image_info, pre_transform=self.pre_transform))

        # Aggregate sample_stats_list
        image_s0_list, image_s1_list, image_s2_list, class_freq_list = zip(*sample_stats_list)
        image_s0_array = np.stack(image_s0_list, axis=0)
        image_s1_array = np.stack(image_s1_list, axis=0)
        image_s2_array = np.stack(image_s2_list, axis=0)
        class_freq_array = np.stack(class_freq_list, axis=0)

        image_s0_total = np.sum(image_s0_array, axis=0)
        image_s1_total = np.sum(image_s1_array, axis=0)
        image_s2_total = np.sum(image_s2_array, axis=0)

        image_mean = image_s1_total / image_s0_total
        image_std = np.sqrt(image_s2_total/image_s0_total - np.power(image_mean, 2))
        class_freq = np.sum(class_freq_array*image_s0_array[:, None], axis=0) / image_s0_total

        # Save aggregated stats
        self.stats = {
            "image_mean": image_mean,
            "image_std": image_std,
            "class_freq": class_freq,
        }
        torch.save(self.stats, self.stats_filepath)

        # Indicates that processing has been performed:
        pathlib.Path(self.processed_flag_filepath).touch()
        
        coco_ds = deepcopy(coco.dataset)
        coco_ds["images"] = image_info_with_pt_file_list
        
        new_annotation_outfile = os.path.join(self.root, f"annotations_{self.fold}_processed.json")
        with open(new_annotation_outfile, 'w') as f_json:
            json.dump(coco_ds, f_json)


    def indices(self):

        return range(len(self))
        
    def get(self, idx):
        image_id = self.image_id_list[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        ffl_pt_path = os.path.join(self.root, image_info["ffl_pt_path"])
        if not os.path.exists(ffl_pt_path):
            raise FileNotFoundError(f"File {ffl_pt_path} not found.")
        data = torch.load(ffl_pt_path, weights_only=False)
        data["image_mean"] = self.stats["image_mean"]
        data["image_std"] = self.stats["image_std"]
        data["class_freq"] = self.stats["class_freq"]
        return data

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            return data
        else:
            return self.index_select(idx)
        
    def add_preprocessed_file_to_coco(self):
        
        image_info_list = []
        coco = self.get_coco()
        for image_id in self.image_id_list:
            image_info = coco.loadImgs(image_id)[0]
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotation_list = coco.loadAnns(annotation_ids)
            image_info["annotation_list"] = annotation_list
            image_info["absolute_img_filepath"] = os.path.join(self.root, image_info["file_name"])
            image_info_list.append(image_info)
            
        

def preprocess_one(image_info, pre_transform):
    data = None
    if os.path.exists(image_info["pt_outfile"]):
        # Load already-processed sample
        try:
            data = torch.load(image_info["pt_outfile"])
        except (EOFError, _pickle.UnpicklingError):
            pass
    if data is None:
        # Process sample:
        image = skimage.io.imread(image_info["absolute_img_filepath"])
        gt_polygons = []
        for annotation in image_info["annotation_list"]:
            flattened_segmentation_list = annotation["segmentation"]
            if len(flattened_segmentation_list) != 1:
                print("WHAT!?!, len(flattened_segmentation_list = {}".format(len(flattened_segmentation_list)))
                print("To implement: if more than one segmentation in flattened_segmentation_list (MS COCO format), does it mean it is a MultiPolygon or a Polygon with holes?")
                raise NotImplementedError
            flattened_arrays = np.array(flattened_segmentation_list)
            coords = np.reshape(flattened_arrays, (-1, 2))
            polygon = shapely.geometry.Polygon(coords)

            # Filter out degenerate polygons (area is lower than 2.0)
            if 2.0 < polygon.area:
                gt_polygons.append(polygon)

        data = {
            "image": image,
            "gt_polygons": gt_polygons,
            "image_relative_filepath": image_info["file_name"],
            "name": image_info["name"],
            "image_id": image_info["id"]
        }
        
        data = pre_transform(data)

        torch.save(data, image_info["pt_outfile"])

    # Compute stats for later aggregation for the whole dataset
    normed_image = data["image"] / 255
    image_s0 = data["image"].shape[0] * data["image"].shape[1]  # Number of pixels
    image_s1 = np.sum(normed_image, axis=(0, 1))  # Sum of pixel normalized values
    image_s2 = np.sum(np.power(normed_image, 2), axis=(0, 1))
    class_freq = np.mean(data["gt_polygons_image"], axis=(0, 1)) / 255

    return image_s0, image_s1, image_s2, class_freq





@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg):

    OmegaConf.resolve(cfg)
    
    fold = "train"
    fold = "val"
    
    dataset = DatasetWithPreprocessing(cfg.dataset.path,
                               pre_transform=data_transforms.get_offline_transform_patch(),
                               fold=fold,
                               pool_size=cfg.num_workers)
    # dataset._process()


    # TODO: need to decide how to integrate this into PPP.
    # the preprocessing could be done here, but now I need to load this data into PPP, i.e. the stored .pt files
    # but if I want to do that with my own dataset structure I need to implement a new Dataset class that loads the .pt files
    # and more importantly, also applies the augmentations
    # all of this is already done inside DatasetWithPreprocessing, so I could also use this class in PPP
    # Note: here in this file, I deleted all the augmentation stuff (online transforms), but should be easy to bring it back
    
    ## Best thing to do would be to add the pt file path to my original coco annotations file so that I could continue to load
    ## with my own dataloader and load the pt file in the __getitem__ method and apply augmentations there.
    ## shouldn't be too hard to augment the frame field

    ###########################################
    ########### DEBUG VISUALIZATION ###########
    ###########################################
    for i in range(len(dataset)):
        print("Images:")
        # Save output to visualize
        seg = np.array(dataset[i]["gt_polygons_image"])
        # seg = np.moveaxis(seg, 0, -1)
        seg_display = utils.get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)
        skimage.io.imsave("gt_seg_edge.png", seg[:, :, 1])

        im = np.array(dataset[i]["image"])
        # im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        gt_crossfield_angle = np.array(dataset[i]["gt_crossfield_angle"])
        # gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
        skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        distances = np.array(dataset[i]["distances"])
        # distances = np.moveaxis(distances, 0, -1)
        distances = 255-(distances*255).astype(np.uint8)
        skimage.io.imsave('distances.png', distances)

        sizes = np.array(dataset[i]["sizes"])
        # sizes = np.moveaxis(sizes, 0, -1)
        sizes = 255-(sizes*255).astype(np.uint8)
        skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(dataset[i]["valid_mask"])
        # # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")
    
if __name__ == '__main__':
    main()
