# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import rasterio
import laspy
import logging
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np
import geopandas as gpd

from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import functional as F
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

from tqdm import tqdm

from ..misc import *

class Predictor:
    def __init__(self, cfg, local_rank=0, world_size=1):
        self.cfg = cfg
        
        self.local_rank = local_rank
        self.world_size = world_size

        self.device = torch.device(f"cuda:{local_rank}")

        verbosity = getattr(logging, self.cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger(self.__class__.__name__, level=verbosity, local_rank=local_rank)

        self.verbosity = verbosity
        self.update_pbar_every = cfg.host.update_pbar_every

        self.logger.log(logging.INFO, f"Init Predictor on rank {local_rank} in world size {world_size}...")
        if self.local_rank == 0 and not os.path.exists(cfg.output_dir):
            self.logger.info(f"Create output directory {cfg.output_dir}")
            os.makedirs(cfg.output_dir, exist_ok=True)
            
        self.is_ddp = self.cfg.host.multi_gpu
        
        self.time_dict = TimeDict()


                           
    def progress_bar(self,item):
        
        disable = self.verbosity >= logging.WARNING
        
        pbar = tqdm(item, total=len(item), 
                      file=sys.stdout, 
                    #   dynamic_ncols=True, 
                      mininterval=self.update_pbar_every,                      
                      disable=disable,
                      position=0,
                      leave=True)
    
        return pbar
    
    def setup_image_size(self, img_res=0.25, img_dim=224):
        self.img_res = img_res
        self.img_dim = img_dim
        
    def load_checkpoint(self):
        
        ## get the file
        checkpoint_file = os.path.join(self.cfg.output_dir, "checkpoints", f"{self.cfg.checkpoint}.pth")
        
        if not os.path.isfile(checkpoint_file):
            raise FileExistsError(f"Checkpoint file {checkpoint_file} not found.")
        self.logger.info(f"Loading model from checkpoint: {checkpoint_file}")
        
        ## load the checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=self.cfg.host.device, weights_only=False)
        for k in checkpoint.keys():
            if "_state_dict" in k:
                checkpoint[k.replace("_state_dict","")] = checkpoint.pop(k)
        
        ## check for correct model type
        cfg = checkpoint.get("cfg",None)
        if cfg is not None:
            if hasattr(cfg, "use_lidar"):
                if not cfg.use_lidar == self.cfg.experiment.encoder.use_lidar:
                    self.logger.error(f"Model checkpoint was trained with use_lidar={cfg.experiment.encoder.use_lidar}, but current config is use_lidar={self.cfg.experiment.encoder.use_lidar}.")
                    raise ValueError("Model checkpoint and current config do not match.")
            if hasattr(cfg, "use_images"):
                if not cfg.use_images == self.cfg.experiment.encoder.use_images:
                    self.logger.error(f"Model checkpoint was trained with use_images={cfg.experiment.encoder.use_images}, but current config is use_images={self.cfg.experiment.encoder.use_images}.")
                    raise ValueError("Model checkpoint and current config do not match.")
            
            if hasattr(cfg, "model.fusion") and isattr(self.cfg.experiment.model, "fusion"):
                if not cfg.experiment.model.fusion == self.cfg.experiment.model.fusion:
                    self.logger.error(f"Model checkpoint was trained with fusion={cfg.experiment.model.fusion}, but current config is fusion={self.cfg.experiment.model.fusion}.")
                    raise ValueError("Model checkpoint and current config do not match.")   
        
        # self.model.load_state_dict(model_state_dict)
        self.model = smart_load_state_dict(self.model, checkpoint["model"], self.logger, strict=True)
        epoch = checkpoint.get("epochs_run",checkpoint.get("epoch",0))
        
        self.logger.info(f"Model loaded from epoch: {epoch}")
        
    def load_image(self, img_infile):
        
        if img_infile is not None:
            
            with rasterio.open(img_infile) as src:
                image = src.read([1, 2, 3])  # shape (3, H, W)
            
            image = torch.from_numpy(image).unsqueeze(0).to(self.cfg.host.device).to(torch.float32)/255.0
            image = F.normalize(image, mean=self.cfg.experiment.encoder.image_mean, std=self.cfg.experiment.encoder.image_std)
            return image
        else:
            return None
        
    
    
    def load_lidar(self, lidar_infile):
        
        img_res = 0.25
        img_dim = 224
        
        if lidar_infile is not None:
            las = laspy.read(lidar_infile)
            lidar = np.vstack((las.x, las.y, las.z)).transpose()
            

            lidar[:, :2] = (lidar[:, :2] - np.min(lidar,axis=0)[:2]) / img_res
            lidar[:, 1] = img_dim - lidar[:, 1]

            # # scale z vals to [0,100]
            scaler = MinMaxScaler(feature_range=(0,self.cfg.experiment.encoder.in_voxel_size.z))
            lidar[:, -1] = scaler.fit_transform(lidar[:, -1].reshape(-1, 1)).squeeze()
            
            lidar = torch.from_numpy(lidar).unsqueeze(0).to(self.cfg.host.device).to(torch.float32).contiguous()
            
            return lidar
        else:
            return None
    

    def load_image_and_tile(self, path,
                            tile_width=224, tile_height=224,
                            downsample_factor=1,
                            georeference=False,
                            out_dir=None,
                            overlap: float = 0.0):
        """
        Load a raster with rasterio, optionally downsample it, and split into tiles.
        Supports optional overlap between tiles in percentage.
        """

        with rasterio.open(path) as src:
            image = src.read([1, 2, 3]) / 255.0
            profile = src.profile
            transform = src.transform

        if not georeference:
            transform = rasterio.Affine(1, 0, 0.0,
                                        0, 1, 0.0)

        # ----- Downsample full image -----
        if downsample_factor > 1:
            C, H, W = image.shape
            self.logger.info(f"Loaded image {path} with shape (C={C}, H={H}, W={W})")
            new_H = H // downsample_factor
            new_W = W // downsample_factor

            image = image[:, :new_H*downsample_factor, :new_W*downsample_factor]
            image = image.reshape(
                C,
                new_H,
                downsample_factor,
                new_W,
                downsample_factor
            ).mean(axis=(2, 4))

        C, H, W = image.shape
        self.logger.info(f"Tilling image {path} with shape (C={C}, H={H}, W={W})")

        tiles = []

        # -------- Overlap math --------
        overlap = float(overlap)
        assert 0.0 <= overlap < 1.0, "overlap must be between 0 and <1"

        x_step = int(tile_width  * (1 - overlap))
        y_step = int(tile_height * (1 - overlap))

        # fallback to avoid zero-step
        if x_step == 0: x_step = 1
        if y_step == 0: y_step = 1

        # compute tile origins in y- and x-direction
        y_positions = list(range(0, H, y_step))
        x_positions = list(range(0, W, x_step))

        # ensure last tile covers boundary
        if y_positions[-1] + tile_height < H:
            y_positions.append(H - tile_height)
        if x_positions[-1] + tile_width < W:
            x_positions.append(W - tile_width)

        # --------- Tiling loop with overlap ---------
        for r, y0 in enumerate(y_positions):
            for c, x0 in enumerate(x_positions):
                y1 = y0 + tile_height
                x1 = x0 + tile_width

                tile = image[:, y0:y1, x0:x1]

                # padding
                pad_h = max(0, tile_height - tile.shape[1])
                pad_w = max(0, tile_width - tile.shape[2])

                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile,
                                ((0, 0), (0, pad_h), (0, pad_w)),
                                mode="constant",
                                constant_values=0)

                # updated affine transform: must use step-based translation
                tile_transform = rasterio.Affine(
                    transform.a,
                    transform.b,
                    transform.c + x0 * transform.a,
                    transform.d,
                    transform.e,
                    transform.f + y0 * transform.e,
                )

                # Save tile if needed
                if out_dir is not None:
                    out_path = f"{out_dir}/tile{r}_{c}.jpeg"
                    out_profile = profile.copy()
                    out_profile.update({
                        "width": tile_width,
                        "height": tile_height,
                        "transform": tile_transform
                    })
                    with rasterio.open(out_path, "w", **out_profile) as dst:
                        dst.write(tile)

                tile_obj = GeoTile(image=tile, transform=tile_transform)
                tiles.append(tile_obj)

        self.logger.info(
            f"Tiled image into {len(tiles)} tiles "
            f"of size (C={C}, H={tile_height}, W={tile_width}) with overlap={overlap}"
        )

        image = np.moveaxis(image, 0, 2)
        return image, tiles
    
    def clip_polygon_with_rect(self, poly, rect):
        """
        Clips a polygon with a rectangle and returns:
        - list of valid Polygon objects
        - or None if the result is empty or invalid
        """
        clipped = poly.intersection(rect)

        if clipped.is_empty:
            return None

        # Case 1: Single polygon
        if isinstance(clipped, Polygon):
            return [clipped] if clipped.is_valid else None

        # Case 2: MultiPolygon
        if isinstance(clipped, MultiPolygon):
            polys = [g for g in clipped.geoms if isinstance(g, Polygon) and g.is_valid]
            return polys if polys else None

        # Case 3: Any other geometry (LineString, GeometryCollection, etc.)
        return None


    def merge_shapely_polygons(self, polygons, tolerance=1.0):
        """
        Merges polygons that are within a certain tolerance using unary_union.
        """
        if not polygons:
            return []

        # # Buffer polygons by tolerance to merge close ones
        # buffered_polys = [poly.buffer(tolerance) for poly in polygons]

        # Merge using unary_union
        merged = unary_union(polygons)

        # If the result is a single polygon, return it in a list
        if isinstance(merged, Polygon):
            return [merged]

        # If the result is a MultiPolygon, return the list of polygons
        if isinstance(merged, MultiPolygon):
            return list(merged.geoms)

        return []

    def tensor_to_shapely_polys(self, 
                                tensor_polygons,
                                transform,
                                overlap_clip=0.0,
                                img_dim=224,  
                                flip_y=False):
        
        shapely_polygons = []
        
        for i,poly in enumerate(tensor_polygons):
            if poly.shape[0] < 3:
                continue
            poly_np = poly.cpu().numpy()
            
            if flip_y:
                poly_np[:,1] = img_dim - poly_np[:,1]
            
            poly_np*=np.array([transform.a,transform.e])
            poly_np+=np.array([transform.c, transform.f])
            
            poly = Polygon(poly_np)
            
            if overlap_clip > 0.0:
                rect = box(56*transform.a+transform.c, 56*transform.a+transform.f, 168*transform.a+transform.c, 168*transform.a+transform.f)   # minx, miny, maxx, maxy
                
                try:
                    poly = self.clip_polygon_with_rect(poly, rect)
                except Exception as e:
                    self.logger.warning(f"Error clipping polygon {i}: {e}")
                    continue
            else:
                if poly.is_valid:
                    poly = [poly]
                else:
                    poly = None
                
            if poly is not None:
                shapely_polygons+=poly

        return shapely_polygons
    

    def export_to_shp(self, shapely_polygons, outfile="polygons.shp", epsg=4326):
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=shapely_polygons)

        # Optionally set a coordinate reference system (CRS), e.g., WGS84
        gdf.set_crs(epsg=epsg, inplace=True)

        # Export to shapefile
        gdf.to_file(outfile)
    
    def plot_prediction(self, polygons, image=None, lidar=None, outfile=None):
        
        if not isinstance(image,np.ndarray):
            image = image.cpu().squeeze().permute(1,2,0).numpy() if image is not None else image
                
        if not len(polygons):
            self.logger.warning(f"No polygons predicted.")
            return
        
        if outfile is None:
            if image is not None:
                name = "image"
            if lidar is not None:
                name = "lidar"
            if image is not None and lidar is not None:
                name = "fusion"
            outfile = f"prediction_{self.cfg.experiment.model.name}_{name}.jpg"
            
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1, 1, figsize=(3*image.shape[0]*px, 3*image.shape[0]*px))
        
        shapely_polygons = []
        for poly in polygons:
            if isinstance(poly, Polygon):
                shapely_polygons.append(poly)
            else:
                shapely_polygons.append(Polygon(poly))
        
        alpha = 1.0
        if image is not None:
            alpha = 0.7
            plot_image(image, ax=ax)
        
        if lidar is not None:
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            plot_point_cloud(lidar, ax=ax, alpha=alpha, pointsize=0.5)
                    
        plot_shapely_polygons(shapely_polygons, ax=ax,pointcolor=[1,1,0],edgecolor=[1,0,1],linewidth=6,pointsize=12)
        
        self.logger.info(f"Save prediction to {outfile}")
        plt.savefig(outfile)