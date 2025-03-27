import os
import pathlib
import warnings
import hydra

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

# from lydorn_utils import print_utils
# from lydorn_utils import python_utils

from torch_lydorn.torch.utils.data import Dataset as LydornDataset, __repr__

from torch_lydorn.torchvision.datasets import utils


import data_transforms

class DatasetPreprocessor(torch.utils.data.Dataset):
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
        self._process()

    def load_image_ids(self):

        coco = self.get_coco()
        image_id_list = coco.getImgIds(catIds=coco.getCatIds())

        return image_id_list

    def get_coco(self):
        if self.coco is None:
            annotation_filename = f"annotations_{self.fold}.json"
            annotations_filepath = os.path.join(self.root, annotation_filename)
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
            print("Dataset already processed. Skipping processing.")
            return

        print('Pre-Processing...')

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        
        print('Done!')

    def process(self):

        image_info_list = []
        coco = self.get_coco()
        for image_id in self.image_id_list:
            image_info = coco.loadImgs(image_id)[0]
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotation_list = coco.loadAnns(annotation_ids)
            image_info["annotation_list"] = annotation_list
            image_info["absolute_img_filepath"] = os.path.join(self.root, image_info["file_name"])
            image_info_list.append(image_info)

        
        # partial_preprocess_one = partial(preprocess_one, pre_transform=self.pre_transform,
        #                                  processed_dir=self.processed_dir)
        # with Pool(self.pool_size) as p:
        #     sample_stats_list = list(tqdm(p.imap(partial_preprocess_one, image_info_list), total=len(image_info_list)))
        sample_stats_list = []
        for image_info in tqdm(image_info_list):
            sample_stats_list.append(preprocess_one(image_info, pre_transform=self.pre_transform, processed_dir=self.processed_dir))

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

    def indices(self):

        return range(len(self))
        
    def get(self, idx):
        image_id = self.image_id_list[idx]
        data = torch.load(os.path.join(self.processed_dir, "data_{:012d}.pt".format(image_id)))
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
        


def preprocess_one(image_info, pre_transform, processed_dir):
    out_filepath = os.path.join(processed_dir, "data_{:012d}.pt".format(image_info["id"]))
    data = None
    if os.path.exists(out_filepath):
        # Load already-processed sample
        try:
            data = torch.load(out_filepath)
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
            "name": os.path.splitext(os.path.basename(image_info["file_name"]))[0],
            "image_id": image_info["id"]
        }
        
        data = pre_transform(data)

        torch.save(data, out_filepath)

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
    
    config = {
        "data_dir_candidates": [
                "/data/titane/user/nigirard/data",
                "~/data",
                "/data"
        ],
        "dataset_params": {
            "small": True,
            "root_dirname": "mapping_challenge_dataset",
            "seed": 0,
            "train_fraction": 0.75
        },
        "num_workers": 1,
        "data_aug_params": {
            "enable": False,
            "vflip": True,
            "affine": True,
            "color_jitter": True,
            "device": "cuda"
        }
    }

    dataset = DatasetPreprocessor(cfg.dataset.path,
                               pre_transform=data_transforms.get_offline_transform_patch(),
                               fold="val",
                               pool_size=config["num_workers"])


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
    
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=10, 
    #                                           collate_fn=torch.utils.data.dataloader.default_collate,
    #                                           shuffle=True, num_workers=config["num_workers"])
    
    
    # for batch in tqdm(data_loader):
    #     print("Images:")
    #     print(batch["image_relative_filepath"])
    #     print(batch["image"].shape)
    #     print(batch["gt_polygons_image"].shape)

    #     print(batch["image"].shape)
    #     print(batch["gt_polygons_image"].shape)

    #     # Save output to visualize
    #     seg = np.array(batch["gt_polygons_image"][0])
    #     seg = np.moveaxis(seg, 0, -1)
    #     seg_display = utils.get_seg_display(seg)
    #     seg_display = (seg_display * 255).astype(np.uint8)
    #     skimage.io.imsave("gt_seg.png", seg_display)
    #     skimage.io.imsave("gt_seg_edge.png", seg[:, :, 1])

    #     im = np.array(batch["image"][0])
    #     im = np.moveaxis(im, 0, -1)
    #     skimage.io.imsave('im.png', im)

    #     gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
    #     gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
    #     skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

    #     distances = np.array(batch["distances"][0])
    #     distances = np.moveaxis(distances, 0, -1)
    #     skimage.io.imsave('distances.png', distances)

    #     sizes = np.array(batch["sizes"][0])
    #     sizes = np.moveaxis(sizes, 0, -1)
    #     skimage.io.imsave('sizes.png', sizes)

    #     # valid_mask = np.array(batch["valid_mask"][0])
    #     # valid_mask = np.moveaxis(valid_mask, 0, -1)
    #     # skimage.io.imsave('valid_mask.png', valid_mask)

    #     input("Press enter to continue...")


if __name__ == '__main__':
    main()
