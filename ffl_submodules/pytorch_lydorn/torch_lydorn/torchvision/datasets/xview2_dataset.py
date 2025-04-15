import fnmatch
import os.path
import pathlib
import random
import sys
import time
from collections import defaultdict

import shapely.geometry
import shapely.wkt
import multiprocessing
import itertools
import skimage.io
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torchvision

from lydorn_utils import run_utils, image_utils, polygon_utils, geo_utils
from lydorn_utils import print_utils
from lydorn_utils import python_utils

from torch_lydorn.torchvision.datasets import utils


class xView2Dataset(torch.utils.data.Dataset):
    """
    xView2 xBD dataset: https://xview2.org/
    """

    def __init__(self, root: str, fold: str = "train", pre_process: bool = True,
                 patch_size: int = None,
                 pre_transform=None, transform=None, small: bool = False, pool_size: int = 1, raw_dirname: str = "raw",
                 processed_dirname: str = "processed"):
        """

        @param root:
        @param fold:
        @param pre_process: If True, the dataset will be pre-processed first, saving training patches on disk. If False, data will be serve on-the-fly without any patching.
        @param patch_size:
        @param pre_transform:
        @param transform:
        @param small: If True, use a small subset of the dataset (for testing)
        @param pool_size:
        @param processed_dirname:
        """
        self.root = root
        self.fold = fold
        self.pre_process = pre_process
        self.patch_size = patch_size
        self.pre_transform = pre_transform
        self.transform = transform
        self.small = small
        if self.small:
            print_utils.print_info("INFO: Using small version of the xView2 xBD dataset.")
        self.pool_size = pool_size
        self.raw_dirname = raw_dirname

        if self.pre_process:
            # Setup of pre-process
            self.processed_dirpath = os.path.join(self.root, processed_dirname, self.fold)
            stats_filepath = os.path.join(self.processed_dirpath, "stats-small.pt" if self.small else "stats.pt")
            processed_relative_paths_filepath = os.path.join(self.processed_dirpath,
                                                        "processed_paths-small.json" if self.small else "processed_paths.json")

            # Check if dataset has finished pre-processing by checking processed_relative_paths_filepath:
            if os.path.exists(processed_relative_paths_filepath):
                # Process done, load stats and processed_relative_paths
                self.stats = torch.load(stats_filepath)
                self.processed_relative_paths = python_utils.load_json(processed_relative_paths_filepath)
            else:
                # Pre-process not finished, launch it:
                tile_info_list = self.get_tile_info_list()
                self.stats = self.process(tile_info_list)
                # Save stats
                torch.save(self.stats, stats_filepath)
                # Save processed_relative_paths
                self.processed_relative_paths = [tile_info["processed_relative_filepath"] for tile_info in tile_info_list]
                python_utils.save_json(processed_relative_paths_filepath, self.processed_relative_paths)
        else:
            # Setup data sample list
            self.tile_info_list = self.get_tile_info_list()

    def get_tile_info_list(self):
        tile_info_list = []
        fold_dirpath = os.path.join(self.root, self.raw_dirname, self.fold)
        images_dirpath = os.path.join(fold_dirpath, "images")
        image_filenames = fnmatch.filter(os.listdir(images_dirpath), "*_pre_disaster.png")
        image_filenames = sorted(image_filenames)
        disaster_samples_dict = defaultdict(int)
        for image_filename in image_filenames:
            name_split = image_filename.split("_")
            disaster = name_split[0]
            if self.small:
                disaster_samples_dict[disaster] += 1
                if 10 < disaster_samples_dict[disaster]:
                    continue  # Skip this sample as there is already enough for the small dataset
            number = int(name_split[1])
            tile_info = {
                "name": f"{disaster}_{number:08d}",
                "disaster": disaster,
                "number": number,
                "image_filepath": os.path.join(fold_dirpath, "images", f"{disaster}_{number:08d}_pre_disaster.png"),
                "label_filepath": os.path.join(fold_dirpath, "labels", f"{disaster}_{number:08d}_pre_disaster.json"),
                "processed_relative_filepath": os.path.join(disaster, f"{number:08d}.pt")
            }
            tile_info_list.append(tile_info)
        return tile_info_list

    def process(self, tile_info_list):
        # os.makedirs(os.path.join(self.root, self.processed_dirname), exist_ok=True)
        with multiprocess.Pool(self.pool_size) as p:
            list_of_stats = list(
                tqdm(p.imap(self._process_one, tile_info_list), total=len(tile_info_list), desc="Process"))

         # Aggregate stats
        mean_per_disaster = defaultdict(list)
        var_per_disaster = defaultdict(list)
        class_freq = []
        for stats in list_of_stats:
            mean_per_disaster[stats["disaster"]].append(stats["mean"])
            var_per_disaster[stats["disaster"]].append(stats["var"])
            class_freq.append(stats["class_freq"])
        stats = {
            "mean": {},
            "std": {},
            "class_freq": None
        }
        for disaster in mean_per_disaster.keys():
            stats["mean"][disaster] = np.mean(np.stack(mean_per_disaster[disaster], axis=0), axis=0)
            stats["std"][disaster] = np.sqrt(np.mean(np.stack(var_per_disaster[disaster], axis=0), axis=0))
        stats["class_freq"] = np.mean(np.stack(class_freq, axis=0), axis=0)

        return stats

    def load_raw_data(self, tile_info):
        # Image:
        tile_info["image"] = skimage.io.imread(tile_info["image_filepath"])
        assert len(tile_info["image"].shape) == 3 and tile_info["image"].shape[
            2] == 3, f"image should have shape (H, W, 3), not {tile_info['image'].shape}..."

        # Annotations:
        label_json = python_utils.load_json(tile_info["label_filepath"])
        features_xy = label_json["features"]["xy"]
        tile_info["gt_polygons"] = [shapely.wkt.loads(obj["wkt"]) for obj in features_xy]

        return tile_info

    def _process_one(self, tile_info):
        # --- Init
        processed_tile_filepath = os.path.join(self.processed_dirpath, tile_info["processed_relative_filepath"])
        processed_tile_dirpath = os.path.dirname(processed_tile_filepath)
        stats_filepath = os.path.join(processed_tile_dirpath, f"{tile_info['number']:08d}.stats.pt")
        os.makedirs(processed_tile_dirpath, exist_ok=True)

        # --- Check if tile has been processed already
        if os.path.exists(stats_filepath):
            stats = torch.load(stats_filepath)
            return stats

        tile_info = self.load_raw_data(tile_info)

        tile_info = self.pre_transform(tile_info)

        # Compute stats
        stats = {
            "mean": np.mean(tile_info["image"].reshape(-1, tile_info["image"].shape[-1]), axis=0) / 255,
            "var": np.var(tile_info["image"].reshape(-1, tile_info["image"].shape[-1]), axis=0) / 255,
            "class_freq": np.mean(tile_info["gt_polygons_image"], axis=(0, 1)) / 255,
            "disaster": tile_info["disaster"]  # Add disaster name to stats for aggregating per disaster
        }

        # Save data
        torch.save(tile_info, processed_tile_filepath)
        torch.save(stats, stats_filepath)

        return stats

    def __len__(self):
        if self.pre_process:
            return len(self.processed_relative_paths)
        else:
            return len(self.tile_info_list)

    def __getitem__(self, idx):
        if self.pre_process:
            filepath = os.path.join(self.processed_dirpath, self.processed_relative_paths[idx])
            data = torch.load(filepath)
            data["image_mean"] = self.stats["mean"][data["disaster"]]
            data["image_std"] = self.stats["std"][data["disaster"]]
            data["class_freq"] = self.stats["class_freq"]
        else:
            tile_info = self.tile_info_list[idx]
            # Load raw data
            data = self.load_raw_data(tile_info)
            raise NotImplementedError("Need to implement mean and std computation")

        # --- Crop to path_size
        height, width, _ = data["image"].shape
        pre_crop_image_norm = data["image"].shape[0] + data["image"].shape[1]
        crop_i = random.randint(0, height - self.patch_size)
        crop_j = random.randint(0, width - self.patch_size)
        data["image"] = data["image"][crop_i:crop_i + self.patch_size, crop_j:crop_j + self.patch_size]
        data["gt_polygons_image"] = data["gt_polygons_image"][crop_i:crop_i + self.patch_size, crop_j:crop_j + self.patch_size]
        data["gt_crossfield_angle"] = data["gt_crossfield_angle"][crop_i:crop_i + self.patch_size, crop_j:crop_j + self.patch_size]
        data["distances"] = data["distances"][crop_i:crop_i + self.patch_size, crop_j:crop_j + self.patch_size]
        data["sizes"] = data["sizes"][crop_i:crop_i + self.patch_size, crop_j:crop_j + self.patch_size]
        post_crop_image_norm = data["image"].shape[0] + data["image"].shape[1]
        # Sizes and distances are affected by cropping because they are relative to the image's norm (height + width).
        # All non-one pixels have to be renormalized:
        size_ratio = pre_crop_image_norm / post_crop_image_norm
        data["distances"][data["distances"] != 1] *= size_ratio
        data["sizes"][data["sizes"] != 1] *= size_ratio
        # ---

        data = self.transform(data)
        return data


def main():
    # Test using transforms from the frame_field_learning project:
    from frame_field_learning import data_transforms

    config = {
        "data_dir_candidates": [
            "/data/titane/user/nigirard/data",
            "~/data",
            "/data"
        ],
        "dataset_params": {
            "root_dirname": "xview2_xbd_dataset",
            "pre_process": True,
            "small": False,
            "data_patch_size": 725,
            "input_patch_size": 512,

            "train_fraction": 0.75
        },
        "num_workers": 8,
        "data_aug_params": {
            "enable": True,
            "vflip": True,
            "affine": True,
            "scaling": [0.9, 1.1],
            "color_jitter": True,
            "device": "cuda"
        }
    }

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    else:
        print_utils.print_info("Using data from {}".format(data_dir))
    root_dir = os.path.join(data_dir, config["dataset_params"]["root_dirname"])

    # --- Transforms: --- #
    # --- pre-processing transform (done once then saved on disk):
    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,
                                                                            augmentations=config["data_aug_params"][
                                                                                "enable"])
    kwargs = {
        "pre_process": config["dataset_params"]["pre_process"],
        "transform": online_cpu_transform,
        "patch_size": config["dataset_params"]["data_patch_size"],
        "pre_transform": data_transforms.get_offline_transform_patch(),
        "small": config["dataset_params"]["small"],
        "pool_size": config["num_workers"],
    }
    # --- --- #
    fold = "train"
    if fold == "train":
        dataset = xView2Dataset(root_dir, fold="train", **kwargs)
    elif fold == "val":
        dataset = xView2Dataset(root_dir, fold="train", **kwargs)
    elif fold == "test":
        dataset = xView2Dataset(root_dir, fold="test", **kwargs)
    else:
        raise NotImplementedError

    print(f"dataset has {len(dataset)} samples.")
    print("# --- Sample 0 --- #")
    sample = dataset[0]
    for key, item in sample.items():
        print("{}: {}".format(key, type(item)))

    print("# --- Samples --- #")
    # for data in tqdm(dataset):
    #     pass

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    print("# --- Batches --- #")
    for batch in tqdm(data_loader):

        # batch["distances"] = batch["distances"].float()
        # batch["sizes"] = batch["sizes"].float()

        # im = np.array(batch["image"][0])
        # im = np.moveaxis(im, 0, -1)
        # skimage.io.imsave('im_before_transform.png', im)
        #
        # distances = np.array(batch["distances"][0])
        # distances = np.moveaxis(distances, 0, -1)
        # skimage.io.imsave('distances_before_transform.png', distances)
        #
        # sizes = np.array(batch["sizes"][0])
        # sizes = np.moveaxis(sizes, 0, -1)
        # skimage.io.imsave('sizes_before_transform.png', sizes)

        print("----")
        print(batch["name"])

        print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        if "gt_polygons_image" in batch:
            print("gt_polygons_image:", batch["gt_polygons_image"].shape, batch["gt_polygons_image"].min().item(),
                  batch["gt_polygons_image"].max().item())
            seg = np.array(batch["gt_polygons_image"][0]) / 255
            seg = np.moveaxis(seg, 0, -1)
            seg_display = utils.get_seg_display(seg)
            seg_display = (seg_display * 255).astype(np.uint8)
            skimage.io.imsave("gt_seg.png", seg_display)

        if "gt_crossfield_angle" in batch:
            print("gt_crossfield_angle:", batch["gt_crossfield_angle"].shape, batch["gt_crossfield_angle"].min().item(),
                  batch["gt_crossfield_angle"].max().item())
            gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
            gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
            skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        if "distances" in batch:
            print("distances:", batch["distances"].shape, batch["distances"].float().min().item(),
                  batch["distances"].float().max().item())
            distances = np.array(batch["distances"][0])
            distances = np.moveaxis(distances, 0, -1)
            skimage.io.imsave('distances.png', distances)

        if "sizes" in batch:
            print("sizes:", batch["sizes"].shape, batch["sizes"].float().min().item(), batch["sizes"].float().max().item())
            sizes = np.array(batch["sizes"][0])
            sizes = np.moveaxis(sizes, 0, -1)
            skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")

        print("Apply online tranform:")
        batch = utils.batch_to_cuda(batch)
        batch = train_online_cuda_transform(batch)
        batch = utils.batch_to_cpu(batch)

        print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
        print("gt_polygons_image:", batch["gt_polygons_image"].shape, batch["gt_polygons_image"].min().item(),
              batch["gt_polygons_image"].max().item())
        print("gt_crossfield_angle:", batch["gt_crossfield_angle"].shape, batch["gt_crossfield_angle"].min().item(),
              batch["gt_crossfield_angle"].max().item())
        # print("distances:", batch["distances"].shape, batch["distances"].min().item(), batch["distances"].max().item())
        # print("sizes:", batch["sizes"].shape, batch["sizes"].min().item(), batch["sizes"].max().item())

        # Save output to visualize
        seg = np.array(batch["gt_polygons_image"][0])
        seg = np.moveaxis(seg, 0, -1)
        seg_display = utils.get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)

        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
        gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
        skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        distances = np.array(batch["distances"][0])
        distances = np.moveaxis(distances, 0, -1)
        skimage.io.imsave('distances.png', distances)

        sizes = np.array(batch["sizes"][0])
        sizes = np.moveaxis(sizes, 0, -1)
        skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")


if __name__ == '__main__':
    main()
