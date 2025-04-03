#!/usr/bin/env python3

import sys, csv, random, glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

import json
import geojson
import rasterio
import rasterio.mask
from rasterio.windows import Window
from pyproj import CRS, Transformer

from skimage.transform import resize

from matplotlib import pyplot as plt

from lydorn_utils import polygon_utils


class BuildingDataset(Dataset):

    def __init__(self, tier=1, show_mode=False, augment=False, small_subset=False, crop_size=1024, resize_size=224,
                 window_random_shift=2048, data_dir="./", baseline_mode=False, transform=None, val=False, val_split=0.1, split_seed=42, sampling_mode="polygons"):
        super().__init__()

        self.crop_size = crop_size
        self.resize_size = resize_size

        self.data_dir = data_dir

        random.seed(42)

        # Load TIF and geojson file names
        # from train_metadata.csv
        self.img_ids_to_tif = dict()
        self.img_ids_to_geojson = dict()
        with open(data_dir + "/train_metadata.csv", 'r') as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            imgs=[]
            for row in csvreader:
                if int(row[3]) <= tier:
                    imgs.append(row)
            random.shuffle(imgs)
            n_train = int(len(imgs)*(1.0-val_split))
            if val:
                imgs = imgs[n_train:]
            else:
                imgs = imgs[:n_train]
            for row in imgs:
                imgid = row[0].split("/")[2]
                self.img_ids_to_tif[imgid] = row[0]
                self.img_ids_to_geojson[imgid] = row[1]

        # Load all geojson files
        print("Loading geojson files")

        self.geojson_data = dict()
        self.all_polygons = []
        self.feat_to_img_id = []
        for ids, geojson_f in tqdm(self.img_ids_to_geojson.items()):
            self.geojson_data[ids] = self._load_geojsonfile(data_dir + "/" + geojson_f)
            self.all_polygons += self.geojson_data[ids]["features"]
            for f in self.geojson_data[ids]["features"]:
                self.feat_to_img_id.append(ids)

        print("Number of training polygons:", len(self.all_polygons))

        # Open all tif files
        # and get corresponding transformers
        print("Opening rasters")
        #self.rasters = dict()
        img_to_transformer_dict = dict()
        for ids, img_f in tqdm(self.img_ids_to_tif.items()):
            with rasterio.open(data_dir + "/" + img_f) as raster:
                img_to_transformer_dict[ids] = Transformer.from_crs(CRS.from_proj4("+proj=latlon"), raster.crs)

        print("Reading means and std")
        self.stats = dict()
        for ids, tifs in self.img_ids_to_tif.items():
            with open(data_dir + "/" + ".".join([tifs.split(".")[0], "tif.stats.json"]), "r") as statfile:
                self.stats[ids] = json.load(statfile)
            

        print("Reprojecting polygons")

        self.img_id_to_polys = dict()
        for k, v in self.img_ids_to_tif.items():
            self.img_id_to_polys[k] = []

        for i in tqdm(range(len(self.all_polygons))):
            self.convert_poly(i, img_to_transformer_dict)
            self.img_id_to_polys[self.feat_to_img_id[i]].append(self.all_polygons[i])

        if small_subset:
            self.all_polygons = self.all_polygons[:200]

        self.show_mode = show_mode
        self.baseline_mode = baseline_mode

        self.aug_transforms = transforms.Compose([transforms.RandomVerticalFlip(),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomAffine((180), (0.2, 0.2), (0.5, 2)),
                                                  transforms.ColorJitter(0.2, 0.2, 0.5, 0.2),
                                                  transforms.Resize((resize_size, resize_size)),
                                                  transforms.ToTensor(),
                                                  # transforms.RandomErasing(),
                                                  transforms.ToPILImage()])

        self.noaug_transforms = transforms.Compose([transforms.Resize((resize_size, resize_size))])

        self.mask_transforms = transforms.Compose([transforms.Resize((resize_size, resize_size)),
                                                   transforms.ToTensor()])

        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])

        self.wow_transforms = transform

        self.augment = augment

        self.window_random_shift = window_random_shift

        self.sampling_mode = sampling_mode

    def __len__(self):
        if self.sampling_mode == "polygons":
            return len(self.all_polygons)
        else:
            return 1000

    def __getitem__(self, i):
        img_id = self.feat_to_img_id[i]

        raster = rasterio.open(self.data_dir + "/" + self.img_ids_to_tif[img_id])

        if self.sampling_mode == "random":
            window = self._get_random_window()
        else:
            window = self._get_window(raster, self.all_polygons[i])

        b1, b2, b3, b4 = raster.read(window=window)
        out_image = np.stack([b1, b2, b3])
        out_image = np.swapaxes(out_image, 0, 2)[:, :, 0:3]


        try:
            img = Image.fromarray(out_image, 'RGB')
        except:
            print(window)
            print(raster.width, raster.height)


        # create rasterized edges
        polygons = []
        for feat in self.img_id_to_polys[img_id]:
            try:
                # polywindow = rasterio.features.geometry_window(raster, [feat])
                polywindow = self._get_window(raster, feat, padded=False)

                if rasterio.windows.intersect(polywindow, window):
                    if feat["geometry"]["type"] == "MultiPolygon":
                        local_poly = []
                        for p in feat["geomtry"]["coordinates"]:
                            local_poly += self._polygon_window_pixel_coords(raster, window, p[0])
                    elif feat["geometry"]["type"] == "Polygon":
                        local_poly = self._polygon_window_pixel_coords(raster, window,
                                                                       feat["geometry"]["coordinates"][0])
                    ratio = self.resize_size / self.crop_size
                    scaled_poly = []
                    for point in local_poly:
                        x, y = point
                        scaled_poly.append((x * ratio, y * ratio))
                    polygons.append(scaled_poly)
            except:
                pass

        masks = polygon_utils.draw_polygons(polygons, (self.resize_size, self.resize_size), line_width=2, antialiasing=True)
        masks = np.moveaxis(np.array(masks), 2, 0)

        angles = polygon_utils.init_angle_field(polygons, (self.resize_size, self.resize_size), line_width=4)
        angles = np.expand_dims(angles, 0)

        if self.augment:
            trans = self.aug_transforms
        else:
            trans = self.noaug_transforms

        mean = [self.stats[img_id]["stats"]["mean_r"], self.stats[img_id]["stats"]["mean_g"], self.stats[img_id]["stats"]["mean_b"]]
        std = [self.stats[img_id]["stats"]["std_r"], self.stats[img_id]["stats"]["std_g"], self.stats[img_id]["stats"]["std_b"]]

        out_img = np.array(trans(img))
        out_img = torch.Tensor(np.moveaxis(out_img, -1, 0))


        if self.show_mode:
            return {"image": trans(img), "gt_polygons_image": masks, "gt_crossfield_angle": angles, "image_mean": mean,
                    "image_std": std}
        elif self.baseline_mode:
            masks = np.moveaxis(np.array(masks), 0,2)
            masks = masks / 255
            return {"image": self.img_transforms(trans(img)), "gt_polygons_image": masks}
        else:
            return {"image": out_img, 
                    "gt_polygons_image": torch.Tensor(masks), 
                    "gt_crossfield_angle": torch.Tensor(angles), 
                    "image_mean": torch.Tensor(mean),
                    "image_std": torch.Tensor(std)}

    def _get_window(self, raster, feat, padded=True):

        poly = feat["geometry"]["coordinates"]

        box = []

        if feat["geometry"]["type"] == "MultiPolygon":
            local_poly = []
            for p in feat["geometry"]["coordinates"]:
                local_poly += self._polygon_pixel_coords(raster, p[0])
        elif feat["geometry"]["type"] == "Polygon":
            local_poly = self._polygon_pixel_coords(raster, feat["geometry"]["coordinates"][0])

        for i in (0, 1):
            res = sorted(local_poly, key=lambda x: x[i])
            box += list((res[0][i], res[-1][i]))

        if padded:
            toplx, toply = (box[0] + box[1]) / 2 - self.crop_size / 2, (box[2] + box[3]) / 2 - self.crop_size / 2

            while True:
                shift_y = random.randint(-self.window_random_shift, self.window_random_shift)
                shift_x = random.randint(-self.window_random_shift, self.window_random_shift)
                win = Window(toplx + shift_x, toply + shift_y, self.crop_size, self.crop_size)

                # check that the window is in the image
                if win.col_off + win.width < raster.width and win.col_off > 0:
                    if win.row_off + win.height < raster.height and win.row_off > 0:
                        break
            return win
        else:
            return Window(box[0], box[1], box[2] - box[0], box[3] - box[1])

    def _get_random_window(self):

        rastern = random.randint(0, len(self.img_ids_to_tif)-1)

        img_id = list(self.img_ids_to_tif)[rastern]

        tiff = self.img_ids_to_tif[img_id]

        raster = rasterio.open(self.data_dir + "/" + tiff)
        
        while True:
            x = random.randint(0, raster.width-1)
            y = random.randint(0, raster.height-1)

            sample = [val for val in raster.sample([raster.transform * (x,y)])][0]
            if sample[3] == 255:
                win = Window(x - self.crop_size/2, y - self.crop_size/2, self.crop_size, self.crop_size)
                if win.col_off + win.width < raster.width and win.col_off > 0:
                    if win.row_off + win.height < raster.height and win.row_off > 0:
                        return win, img_id


    def _polygon_pixel_coords(self, raster, poly):
        # converts to raster pixel coords from utm
        local_poly = []
        for point in poly:
            local_poly.append(~raster.transform * (point[0], point[1]))

        return local_poly

    def _polygon_window_pixel_coords(self, raster, window, poly):
        # converts to window coords from utm
        local_poly = []
        for point in poly:
            local_poly.append(~raster.window_transform(window) * (point[0], point[1]))

        return local_poly

    def _load_geojsonfile(self, filename):
        with open(filename, 'r') as f:
            content = f.read()
            json = geojson.loads(content)
        return json

    def convert_poly(self, i, transformers):
        img_id = self.feat_to_img_id[i]
        transformer = transformers[img_id]

        geom = self.all_polygons[i]["geometry"]

        if geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for part in poly:
                    for point in part:
                        point[0], point[1] = list(transformer.transform(point[0], point[1]))
        elif geom["type"] == "Polygon":
            for poly in geom["coordinates"]:
                for point in poly:
                    point[0], point[1] = list(transformer.transform(point[0], point[1]))

    def augmentation(self, enable=True):
        self.augment = enable

class RasterizedOpenCities(BuildingDataset):

    def __init__(self, 
                tier=1, 
                show_mode=False, 
                augment=False, 
                small_subset=False, 
                crop_size=1024, 
                resize_size=224,
                window_random_shift=2048, 
                data_dir="./", 
                baseline_mode=False, 
                transform=None, 
                val=False, 
                val_split=0.1, 
                split_seed=42,
                sampling_mode="polygons"):

        super().__init__(tier, show_mode, augment, small_subset, crop_size,
                         resize_size, window_random_shift, data_dir, baseline_mode, transform, val, val_split, split_seed, sampling_mode)

        self.img_ids_to_label_raster = dict()
        with open(data_dir + "/train_metadata.csv", 'r') as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            for row in csvreader:
                if int(row[3]) <= tier:
                    imgid = row[0].split("/")[2]
                    self.img_ids_to_label_raster[imgid] = row[1][:-7] + "tif"

    def __getitem__(self, i):


        if self.sampling_mode == "random":
            window, raster_id = self._get_random_window()
            img_id=raster_id
            raster = rasterio.open(self.data_dir + "/" + self.img_ids_to_tif[raster_id])
            label_raster = rasterio.open(self.data_dir + "/" +
                                     self.img_ids_to_label_raster[raster_id])
        else:
            img_id = self.feat_to_img_id[i]
            raster = rasterio.open(self.data_dir + "/" + self.img_ids_to_tif[img_id])
            label_raster = rasterio.open(self.data_dir + "/" +
                                     self.img_ids_to_label_raster[img_id])
            window = self._get_window(raster, self.all_polygons[i])

        b1, b2, b3, b4 = raster.read(window=window)
        out_image = np.stack([b1, b2, b3])
        out_image = np.moveaxis(out_image, 0, 2)[:, :, 0:3]
        img = Image.fromarray(out_image, 'RGB')

        if self.augment:
            trans = self.aug_transforms
        else:
            trans = self.noaug_transforms

        fill, edges, vertices, angles = label_raster.read(window=window)
        masks = np.stack([fill, edges, vertices])

        mean = [self.stats[img_id]["stats"]["mean_r"], self.stats[img_id]["stats"]["mean_g"], self.stats[img_id]["stats"]["mean_b"]]
        std = [self.stats[img_id]["stats"]["std_r"], self.stats[img_id]["stats"]["std_g"], self.stats[img_id]["stats"]["std_b"]]

        masks = np.moveaxis(masks, 0, 2)
        masks = Image.fromarray(masks, "RGB")
        masks = masks.resize((self.resize_size, self.resize_size))
        angles = Image.fromarray(angles)
        angles = angles.resize((self.resize_size, self.resize_size))

        if self.show_mode:
            masks = np.moveaxis(np.array(masks), 0, 2)
            return {"image": trans(img), "gt_polygons_image": masks, "gt_crossfield_angle": angles, "image_mean": mean,
                    "image_std": std}
        elif self.baseline_mode:
            masks = np.array(masks, dtype=np.float) / 255
            return {"image": self.img_transforms(trans(img)), "gt_polygons_image": masks}
        else:
            out_img = np.array(trans(img))
            out_img = torch.Tensor(np.moveaxis(out_img, -1, 0))
            masks = np.moveaxis(np.array(masks), 2, 0)
            masks = torch.Tensor(np.array(masks))
            angles = np.expand_dims(angles, 0)
            angles = torch.Tensor(np.array(angles))
            return {"image": out_img, 
                    "gt_polygons_image": masks, 
                    "gt_crossfield_angle": angles, 
                    "image_mean": torch.Tensor(mean),
                    "image_std": torch.Tensor(std),
                    "name":str(i),
                    "original_image": img_id}


class OpenCitiesTestDataset(Dataset):

    def __init__(self, img_dir, output_size):
        super().__init__()
        self.tif_files = glob.glob(img_dir + "/*/*.tif")
        print("Found", str(len(self.tif_files)), "test images")

        self.transforms = transforms.Compose([transforms.Resize((output_size, output_size))])

    def __len__(self):
        return len(self.tif_files)

    def __getitem__(self, i):
        img = self.get_img(i)
        imgscaled = img / 255.0
        return {"image": img, 
                "image_filepath": self.tif_files[i],
                "name":self.get_id(i), 
                "image_mean": torch.mean(imgscaled, (1,2)), 
                "image_std" : torch.std(imgscaled, (1,2))}

    def get_id(self, i):

        path = self.tif_files[i]
        return path.split("/")[-2]

    def get_img(self, i):

        img = Image.open(self.tif_files[i])
        img = img.convert("RGB")
        img = np.moveaxis(np.array(self.transforms(img)),2,0)

        return torch.Tensor(img)




if __name__ == "__main__":

    ds = RasterizedOpenCities(show_mode=True, tier=1, small_subset=False, sampling_mode="random")

    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    for i in range(n_samples):
        break
        n = random.randint(0, len(ds))

        sample = ds[n]

        fig = plt.figure()

        fig.add_subplot(1, 3, 1)
        plt.imshow(sample["image"])
        fig.add_subplot(1, 3, 2)
        plt.imshow(sample["gt_polygons_image"])
        fig.add_subplot(1, 3, 3)
        plt.imshow(sample["gt_crossfield_angle"])
        plt.show()

    print("Testing whole dataset")
    for i in tqdm(range(len(ds))):
        sample = ds[i]
