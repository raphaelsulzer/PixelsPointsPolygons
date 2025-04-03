import skimage.io
from functools import partial
from multiprocess import Pool
import itertools

import numpy as np
import rasterio
import rasterio.rio.insp
import fiona
from pyproj import Proj, transform, CRS

import torch_lydorn.torch.utils.data

import os

from tqdm import tqdm

import torch

from frame_field_learning import data_transforms

from lydorn_utils import run_utils
from lydorn_utils import print_utils
from lydorn_utils import python_utils
from lydorn_utils import polygon_utils
from lydorn_utils import ogr2ogr
from lydorn_utils import image_utils


class LuxcartaBuildings(torch_lydorn.torch.utils.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, fold="train", patch_size=None, patch_stride=None,
                 pool_size=1):
        assert fold in {"train", "test"}, "fold should be either train of test"
        self.fold = fold
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.pool_size = pool_size
        self._processed_filepaths = None
        # TODO: implement pool_size option
        super(LuxcartaBuildings, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw', self.fold)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.fold)

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_sample_metadata_list(self):
        returned_list = []
        for dirname in os.listdir(self.raw_dir):
            dirpath = os.path.join(self.raw_dir, dirname)
            if os.path.isdir(dirpath):
                image_filepath_list = python_utils.get_filepaths(dirpath, endswith_str="_crop.tif",
                                                                 startswith_str="Ortho",
                                                                 not_endswith_str=".tif_crop.tif")
                gt_polygons_filepath_list = python_utils.get_filepaths(dirpath, endswith_str=".shp",
                                                                       startswith_str="Building")
                mask_filepath_list = python_utils.get_filepaths(dirpath, endswith_str=".kml", startswith_str="zone")
                if len(image_filepath_list) and len(gt_polygons_filepath_list):
                    metadata = {
                        "dirname": dirname,
                        "image_filepath": image_filepath_list[0],
                        "gt_polygons_filepath": gt_polygons_filepath_list[0],
                    }
                    if len(mask_filepath_list):
                        metadata["mask_filepath"] = mask_filepath_list[0]
                    returned_list.append(metadata)
        return returned_list

    @property
    def processed_filepaths(self):
        if self._processed_filepaths is None:
            self._processed_filepaths = []
            for dirname in os.listdir(self.processed_dir):
                dirpath = os.path.join(self.processed_dir, dirname)
                if os.path.isdir(dirpath):
                    filepath_list = python_utils.get_filepaths(dirpath, endswith_str=".pt", startswith_str="data.")
                    self._processed_filepaths.extend(filepath_list)
        return self._processed_filepaths

    def __len__(self):
        return len(self.processed_filepaths)

    def _download(self):
        pass

    def download(self):
        pass

    def _get_mask_multi_polygon(self, mask_filepath, shp_srs):
        # Read mask and convert to shapefile's crs:
        mask_shp_filepath = os.path.join(os.path.dirname(mask_filepath), "mask.shp")

        # TODO: see what's up with the warnings:
        ogr2ogr.main(["", "-f", "ESRI Shapefile", mask_shp_filepath,
                      mask_filepath])  # Convert .kml into .shp
        mask = fiona.open(mask_shp_filepath, "r")
        mask_srs = Proj(mask.crs["init"])

        mask_multi_polygon = []
        for feat in mask:
            mask_polygon = []
            for point in feat['geometry']['coordinates'][0]:
                long, lat = point[:2]  # one 2D point of the LinearRing
                x, y = transform(mask_srs, shp_srs, long, lat, always_xy=True)  # transform the point
                mask_polygon.append((x, y))
            mask_multi_polygon.append(mask_polygon)
            # mask_polygon is now in UTM proj, ready to compare to shapefile proj

        mask.close()

        return mask_multi_polygon

    def _read_annotations(self, annotations_filepath, raster, mask_filepath=None):
        shapefile = fiona.open(annotations_filepath, "r")
        shp_srs = Proj(shapefile.crs["init"])
        raster_srs = Proj(raster.crs)

        # --- Read and crop shapefile with mask if specified --- #
        if mask_filepath is not None:
            mask_multi_polygon = self._get_mask_multi_polygon(mask_filepath, shp_srs)
        else:
            mask_multi_polygon = None

        process_feat_partial = partial(process_feat, shp_srs=shp_srs, raster_srs=raster_srs,
                                       raster_transform=raster.transform, mask_multi_polygon=mask_multi_polygon)
        with Pool() as pool:
            out_polygons = list(
                tqdm(pool.imap(process_feat_partial, shapefile), desc="Process shp feature", total=len(shapefile),
                     leave=True))
        out_polygons = list(itertools.chain.from_iterable(out_polygons))

        shapefile.close()

        return out_polygons

    def process(self, metadata_list):
        progress_bar = tqdm(metadata_list, desc="Pre-process")
        for metadata in progress_bar:
            progress_bar.set_postfix(image=metadata["dirname"], status="Loading image")
            # Load image
            # image = skimage.io.imread(metadata["image_filepath"])
            raster = rasterio.open(metadata["image_filepath"])
            # print(raster)
            # print(dir(raster))
            # print(raster.meta)
            # exit()

            progress_bar.set_postfix(image=metadata["dirname"], status="Process shapefile")
            mask_filepath = metadata["mask_filepath"] if "mask_filepath" in metadata else None
            gt_polygons = self._read_annotations(metadata["gt_polygons_filepath"], raster, mask_filepath=mask_filepath)

            # Compute image mean and std
            b1, b2, b3 = raster.read()
            image = np.stack([b1, b2, b3], axis=-1)
            progress_bar.set_postfix(image=metadata["dirname"], status="Compute mean and std")
            image_float = image / 255
            mean = np.mean(image_float.reshape(-1, image_float.shape[-1]), axis=0)
            std = np.std(image_float.reshape(-1, image_float.shape[-1]), axis=0)

            if self.patch_size is not None:
                # Patch the tile
                progress_bar.set_postfix(image=metadata["dirname"], status="Patching")
                patch_stride = self.patch_stride if self.patch_stride is not None else self.patch_size
                patch_boundingboxes = image_utils.compute_patch_boundingboxes(image.shape[0:2],
                                                                              stride=patch_stride,
                                                                              patch_res=self.patch_size)
                for i, patch_boundingbox in enumerate(tqdm(patch_boundingboxes, desc="Process patches", leave=False)):
                    patch_gt_polygons = polygon_utils.crop_polygons_to_patch_if_touch(gt_polygons, patch_boundingbox)
                    if len(patch_gt_polygons) == 0:
                        # Do not save patches empty of polygons # TODO: keep empty patches?
                        break
                    patch_image = image[patch_boundingbox[0]:patch_boundingbox[2],
                                  patch_boundingbox[1]:patch_boundingbox[3], :]
                    sample = {
                        "dirname": metadata["dirname"],
                        "image": patch_image,
                        "image_mean": mean,
                        "image_std": std,
                        "gt_polygons": patch_gt_polygons,
                        "image_filepath": metadata["image_filepath"],
                    }

                    if self.pre_transform:
                        sample = self.pre_transform(sample)

                    filepath = os.path.join(self.processed_dir, sample["dirname"], "data.{:06d}.pt".format(i))
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    torch.save(sample, filepath)
            else:
                # Tile is saved as is
                sample = {
                    "dirname": metadata["dirname"],
                    "image": image,
                    "image_mean": mean,
                    "image_std": std,
                    "gt_polygons": gt_polygons,
                    "image_filepath": metadata["image_filepath"],
                }

                if self.pre_transform:
                    sample = self.pre_transform(sample)

                filepath = os.path.join(self.processed_dir, sample["dirname"], "data.{:06d}.pt".format(0))
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                torch.save(sample, filepath)

            flag_filepath = os.path.join(self.processed_dir, metadata["dirname"], "flag.json")
            flag = python_utils.load_json(flag_filepath)
            flag["done"] = True
            python_utils.save_json(flag_filepath, flag)

            raster.close()

    def _process(self):
        to_process_metadata_list = []
        for metadata in self.raw_sample_metadata_list:
            flag_filepath = os.path.join(self.processed_dir, metadata["dirname"], "flag.json")
            if os.path.exists(flag_filepath):
                flag = python_utils.load_json(flag_filepath)
                if not flag["done"]:
                    to_process_metadata_list.append(metadata)
            else:
                flag = {
                    "done": False
                }
                python_utils.save_json(flag_filepath, flag)
                to_process_metadata_list.append(metadata)

        if len(to_process_metadata_list) == 0:
            return

        print('Processing...')

        torch_lydorn.torch.utils.data.makedirs(self.processed_dir)
        self.process(to_process_metadata_list)

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        torch.save(torch_lydorn.torch.utils.data.__repr__(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        torch.save(torch_lydorn.torch.utils.data.__repr__(self.pre_filter), path)

        print('Done!')

    def get(self, idx):
        filepath = self.processed_filepaths[idx]
        data = torch.load(filepath)
        data["patch_bbox"] = torch.tensor([0, 0, 0, 0])  # TODO: implement in pre-processing
        tile_name = os.path.basename(os.path.dirname(filepath))
        patch_name = os.path.basename(filepath)
        patch_name = patch_name[len("data."):-len(".pt")]
        data["name"] = tile_name + "." + patch_name
        return data


def process_feat(feat, shp_srs, raster_srs, raster_transform, mask_multi_polygon=None):
    out_polygons = []

    if feat['geometry']["type"] == "Polygon":
        poly = feat['geometry']['coordinates']
        polygons = process_polygon_feat(poly, shp_srs, raster_srs, raster_transform, mask_multi_polygon)
        out_polygons.extend(polygons)

    elif feat['geometry']["type"] == "MultiPolygon":
        for poly in feat['geometry']["coordinates"]:
            polygons = process_polygon_feat(poly, shp_srs, raster_srs, raster_transform, mask_multi_polygon)
            out_polygons.extend(polygons)

    return out_polygons


def process_polygon_feat(in_polygon, shp_srs, raster_srs, raster_transform, mask_multi_polygon=None):
    out_polygons = []
    points = in_polygon[0]  # TODO: handle holes
    # Intersect with mask_polygon if specified
    if mask_multi_polygon is not None:
        multi_polygon_simple = polygon_utils.intersect_polygons(points, mask_multi_polygon)
        if multi_polygon_simple is None:
            return out_polygons
    else:
        multi_polygon_simple = [points]

    for polygon_simple in multi_polygon_simple:
        new_poly = []
        for point in polygon_simple:
            x, y = point[:2]  # 740524.429227941 7175355.263524155
            x, y = transform(shp_srs, raster_srs, x, y)  # transform the point # 740520.728530676 7175320.732711278
            j, i = ~raster_transform * (x, y)
            new_poly.append((i, j))  # 2962.534577447921 2457.457061359659
        out_polygons.append(np.array(new_poly))

    return out_polygons


def get_seg_display(seg):
    seg_display = np.zeros([seg.shape[0], seg.shape[1], 4], dtype=np.float)
    if len(seg.shape) == 2:
        seg_display[..., 0] = seg
        seg_display[..., 3] = seg
    else:
        for i in range(seg.shape[-1]):
            seg_display[..., i] = seg[..., i]
        seg_display[..., 3] = np.clip(np.sum(seg, axis=-1), 0, 1)
    return seg_display


def main():
    # --- Params --- #
    config_name = "config.luxcarta_dataset"
    # --- --- #

    # Load config
    config = run_utils.load_config(config_name)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        exit()

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
    train_pre_transform = data_transforms.get_offline_transform(config,
                                                                augmentations=config["data_aug_params"]["enable"])
    eval_pre_transform = data_transforms.get_offline_transform(config, augmentations=False)
    # --- Online transform done on the host (CPU):
    train_online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                          augmentations=config["data_aug_params"][
                                                                              "enable"])
    eval_online_cpu_transform = data_transforms.get_online_cpu_transform(config, augmentations=False)
    # --- Online transform performed on the device (GPU):
    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,
                                                                            augmentations=config["data_aug_params"][
                                                                                "enable"])
    eval_online_cuda_transform = data_transforms.get_online_cuda_transform(config, augmentations=False)
    # --- --- #

    data_patch_size = config["dataset_params"]["data_patch_size"] if config["data_aug_params"]["enable"] else config["dataset_params"]["input_patch_size"]
    fold = "test"
    if fold == "train":
        dataset = LuxcartaBuildings(root_dir,
                                    transform=train_online_cpu_transform,
                                    patch_size=data_patch_size,
                                    patch_stride=config["dataset_params"]["input_patch_size"],
                                    pre_transform=data_transforms.get_offline_transform_patch(),
                                    fold="train",
                                    pool_size=config["num_workers"])
    elif fold == "test":
        dataset = LuxcartaBuildings(root_dir,
                                    transform=train_online_cpu_transform,
                                    pre_transform=data_transforms.get_offline_transform_patch(),
                                    fold="test",
                                    pool_size=config["num_workers"])

    print("# --- Sample 0 --- #")
    sample = dataset[0]
    print(sample["image"].shape)
    print(sample["gt_polygons_image"].shape)
    print("# --- Samples --- #")
    # for data in tqdm(dataset):
    #     pass

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    print("# --- Batches --- #")
    for batch in tqdm(data_loader):
        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        # Save output to visualize
        seg = np.array(batch["gt_polygons_image"][0]) / 255  # First batch
        seg = np.moveaxis(seg, 0, -1)
        seg_display = get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)

        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        input("Enter to continue...")


if __name__ == '__main__':
    main()
