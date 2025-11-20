# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import laspy
import torch
import numpy as np
import geopandas as gpd
import rasterio

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .predictor_pix2poly import Pix2PolyPredictor

from ..datasets import get_train_loader, get_val_loader, get_test_loader
from ..misc import GeoTile

class Pix2PolyGeoPredictor(Pix2PolyPredictor):
    
    
    def tile_lidar_to_224(self, img=None, las=None, overlap_pct=0.2):
        """
        Split a LAS point cloud into 56m × 56m tiles with optional overlap.

        Args:
            las (laspy.LasData): Input point cloud
            overlap_pct (float): Percentage overlap between tiles (0.0–0.9)
                                Example: 0.25 → 25% overlap.

        Returns:
            list[Tile]: A list of Tile objects (jagged)
        """

        # --- Basic parameters ---
        tile_size = self.img_res * self.img_dim     # 56 m
        overlap_pct = float(overlap_pct)
        assert 0.0 <= overlap_pct < 1.0

        stride = tile_size * (1 - overlap_pct)       # effective step size between tile origins

        # Extract coordinates (M, 3)
        xyz = np.vstack((las.x, las.y, las.z)).T

        # Bounding box
        min_x, min_y, _ = las.header.min
        max_x, max_y, _ = las.header.max

        # --- Determine tile origins based on stride ---
        x_starts = np.arange(min_x, max_x, stride)
        y_starts = np.arange(min_y, max_y, stride)


        # Pre-scaler
        scaler = MinMaxScaler(
            feature_range=(0, self.cfg.experiment.encoder.in_voxel_size.z)
        )

        tiles = []

        # Loop over tile grid
        for tile_min_y in y_starts:
            for tile_min_x in x_starts:

                # Calculate tile bounding box
                tile_max_x = tile_min_x + tile_size
                tile_max_y = tile_min_y + tile_size

                # Select points in this tile (points may appear in multiple tiles!)
                mask = (
                    (xyz[:, 0] >= tile_min_x) & (xyz[:, 0] < tile_max_x) &
                    (xyz[:, 1] >= tile_min_y) & (xyz[:, 1] < tile_max_y)
                )
                pts = xyz[mask]

                if pts.shape[0] == 0:
                    continue

                # --- Local normalization ---
                translation = -np.array([tile_min_x, tile_min_y])
                pts = pts.copy()
                pts[:, :2] = pts[:, :2] + translation       # move to local tile frame
                pts[:, :2] = pts[:, :2] / self.img_res      # scale to 224×224 grid
                pts[:, 2] = scaler.fit_transform(pts[:, 2].reshape(-1, 1)).squeeze()

                # Assertions
                assert np.all((pts[:, 0] >= 0) & (pts[:, 0] <= self.img_dim)), \
                        "X coordinates out of bounds after tiling."
                assert np.all((pts[:, 1] >= 0) & (pts[:, 1] <= self.img_dim)), \
                        "Y coordinates out of bounds after tiling."

                # Append tile
                tiles.append(GeoTile(lidar=pts, translation=translation))

        return tiles

    
    def export_to_shp(self, shapely_polygons, outfile="polygons.shp", epsg=4326):
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=shapely_polygons)

        # Optionally set a coordinate reference system (CRS), e.g., WGS84
        gdf.set_crs(epsg=epsg, inplace=True)

        # Export to shapefile
        gdf.to_file(outfile)
    
    
    def predict_geofile(self,img_infile=None,lidar_infile=None,outfile="polygons.shp"):
        
        self.setup_image_size()
        self.setup_model()
        self.load_checkpoint()
        
        las = laspy.read(lidar_infile)
        
        tiles = self.tile_lidar_to_224(las=las)
                
        batch_size = self.cfg.run_type.batch_size
        iters = len(tiles)//batch_size + int(len(tiles)%batch_size>0)
        
        batch_polygons = []
        
        for i in tqdm(range(iters),desc="Predict batches"):
            
            batch_start = i*batch_size
            batch_end = min((i+1)*batch_size, len(tiles))
            lidar_batch = []

            for j in range(batch_start, batch_end):
                
                lidar_batch.append(torch.from_numpy(tiles[j].lidar).float())

            lidar_batch = torch.nested.nested_tensor(lidar_batch, layout=torch.jagged).to(self.device)
            batch_polygons += self.batch_to_polygons(None, lidar_batch, self.model, self.tokenizer)

        assert(len(batch_polygons) == len(tiles)), f"Number of predicted polygon sets ({len(batch_polygons)}) does not match number of tiles ({len(tiles)})."

        shapely_polygons = []
        for i in range(len(tiles)):
            shapely_polygons += self.tensor_to_shapely_polys(batch_polygons[i],transform=-tiles[i].transform)

        self.export_to_shp(shapely_polygons,outfile=outfile,epsg=las.header.parse_crs().to_epsg())
     
    

    def predict_dataset_to_shp(self, split="val", outfile="./polygon_predictions/out.shp"):
        
        self.setup_model()
        self.load_checkpoint()

        if split == "train":
            loader = get_train_loader(self.cfg,tokenizer=self.tokenizer,logger=self.logger)
        elif split == "val":
            loader = get_val_loader(self.cfg,tokenizer=self.tokenizer,logger=self.logger)
        elif split == "test":
            loader = get_test_loader(self.cfg,tokenizer=self.tokenizer,logger=self.logger)
        else:   
            raise ValueError(f"Unknown split {split}.")
        
        self.logger.info(f"Predicting on {len(loader)} batches...")
        
        batch_polygons = []
        img_infos = []

        for x_image, x_lidar, y_sequence, y_perm, image_ids in self.progress_bar(loader):
            
            if self.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self.device, non_blocking=True)
            
            batch_polygons+= self.batch_to_polygons(x_image, x_lidar, self.model, self.tokenizer)
            img_infos+= loader.dataset.coco.loadImgs(np.atleast_1d(image_ids.squeeze().cpu().numpy()))
            
        shapely_polygons = []

        for i in range(len(batch_polygons)):
            transform = transform = rasterio.Affine(0.25, 0, img_infos[i]['top_left'][0],
                                        0, -0.25, img_infos[i]['top_left'][1])
            shapely_polygons += self.tensor_to_shapely_polys(batch_polygons[i], transform=transform, flip_y=True)

        self.export_to_shp(shapely_polygons,outfile=outfile,epsg=2056)
        
