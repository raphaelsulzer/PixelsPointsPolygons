# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import laspy
import torch
import numpy as np
import geopandas as gpd

from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
from tqdm import tqdm

from .predictor_pix2poly import Pix2PolyPredictor

from ..datasets import get_train_loader, get_val_loader, get_test_loader
class Tile:
    def __init__(self, image=None, lidar=None, translation=None, scale=0.25):
        self.image = image  # Tensor of shape (C, H, W)
        self.lidar = lidar  # Tensor of shape (N, 3)
        self.translation = translation  # Affine transform for georeferencing
        self.scale = scale


class Pix2PolyGeoPredictor(Pix2PolyPredictor):
    
    def setup_image_size(self, img_res=0.25, img_dim=224):
        self.img_res = img_res
        self.img_dim = img_dim


    def tile_input(self, img=None, las=None):
        """
        Split a LAS point cloud into 56m x 56m tiles and return a jagged tensor.

        Args:
            las: laspy.LasData object
            tile_size (float): Tile width and height in meters (default 56).
            device (str): Torch device ('cpu' or 'cuda').

        Returns:
            torch.nested.nested_tensor: shape (B, Ni, 3) with jagged layout
        """
        
        tile_size = self.img_res * self.img_dim  # 56.0 meters
        
        # Extract coordinates as (M, 3)
        xyz = np.vstack((las.x, las.y, las.z)).T

        # Bounding box
        min_x, min_y, _ = las.header.min
        max_x, max_y, _ = las.header.max

        # Compute grid dimensions
        num_tiles_x = int(np.ceil((max_x - min_x) / tile_size))
        num_tiles_y = int(np.ceil((max_y - min_y) / tile_size))

        # Compute tile indices per point
        ix = np.floor((xyz[:, 0] - min_x) / tile_size).astype(int)
        iy = np.floor((xyz[:, 1] - min_y) / tile_size).astype(int)
        tile_ids = iy * num_tiles_x + ix

        scaler = MinMaxScaler(feature_range=(0,self.cfg.experiment.encoder.in_voxel_size.z))

        # Group points per tile
        tiles = []
        for tid in np.unique(tile_ids):
            pts = xyz[tile_ids == tid]
            
            # Recover ix, iy from tile ID
            iy_tile = tid // num_tiles_x
            ix_tile = tid % num_tiles_x

            tile_min_x = min_x + ix_tile * tile_size
            tile_min_y = min_y + iy_tile * tile_size
            
            translation = -np.array([tile_min_x, tile_min_y])
            pts[:,:2] = pts[:,:2]+translation  # Translate points to tile local coords
            pts[:,:2] = pts[:,:2]/self.img_res  # Scale to 224x224x100 grid
            pts[:, -1] = scaler.fit_transform(pts[:, -1].reshape(-1, 1)).squeeze()
            
            assert np.all(pts[:,0] >=0) and np.all(pts[:,0] <= self.img_dim), "X coordinates out of bounds after tiling."
            assert np.all(pts[:,1] >=0) and np.all(pts[:,1] <= self.img_dim), "Y coordinates out of bounds after tiling."
            
            t = Tile(lidar=pts,translation=translation)
            tiles.append(t)

        return tiles
    
    
    
    def tensor_to_shapely_polys(self, tensor_polygons, translation, flip_y=False):
        
        shapely_polygons = []
        
        for i,poly in enumerate(tensor_polygons):
            if poly.shape[0] < 3:
                continue
            poly_np = poly.cpu().numpy()
            
            if flip_y:
                poly_np[:,1] = self.img_dim - poly_np[:,1]
            
            poly_np*=self.img_res
            poly_np+=translation
            
            shapely_polygons.append(Polygon(poly_np))

        return shapely_polygons
    
    
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
        
        batch_size = self.cfg.run_type.batch_size
        
        las = laspy.read(lidar_infile)
        
        tiles = self.tile_input(las=las)
                
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
            shapely_polygons += self.tensor_to_shapely_polys(batch_polygons[i],translation=-tiles[i].translation)

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
            img_infos+= loader.dataset.coco.loadImgs(image_ids.squeeze().cpu().numpy())
            
        self.setup_image_size(img_res=img_infos[0]['res_x'], img_dim=img_infos[0]['width'])
        shapely_polygons = []
        for i in range(len(batch_polygons)):
            shapely_polygons += self.tensor_to_shapely_polys(batch_polygons[i], translation=img_infos[i]['top_left'], flip_y=True)

        self.export_to_shp(shapely_polygons,outfile=outfile,epsg=2056)
        
