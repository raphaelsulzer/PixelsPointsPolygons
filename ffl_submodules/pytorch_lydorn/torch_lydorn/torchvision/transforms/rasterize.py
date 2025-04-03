import math
import sys
import time

import skimage.morphology
import skimage.io
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import shapely.geometry
import shapely.affinity
from lydorn_utils import print_utils
from scipy.ndimage.morphology import distance_transform_edt
import cv2 as cv

from functools import partial

import torch_lydorn.torchvision


class Rasterize(object):
    """Rasterize polygons"""

    def __init__(self, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False, return_distances=False,
                 return_sizes=False):
        self.fill = fill
        self.edges = edges
        self.vertices = vertices
        self.line_width = line_width
        self.antialiasing = antialiasing

        if not return_distances and not return_sizes:
            self.raster_func = partial(draw_polygons, fill=self.fill, edges=self.edges, vertices=self.vertices,
                                       line_width=self.line_width, antialiasing=self.antialiasing)
        elif return_distances and return_sizes:
            self.raster_func = partial(compute_raster_distances_sizes, fill=self.fill, edges=self.edges, vertices=self.vertices,
                                       line_width=self.line_width, antialiasing=self.antialiasing)
        else:
            raise NotImplementedError

    def __call__(self, image, polygons):
        """
        If distances is True, also returns distances image
        (sum of distance to closest and second-closest annotation for each pixel).
        Same for sizes (size of annotation the pixel belongs to).

        """
        size = (image.shape[0], image.shape[1])
        out = self.raster_func(polygons, size)
        return out


def compute_raster_distances_sizes(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    """
    Returns:
         - distances: sum of distance to closest and second-closest annotation for each pixel.
         - size_weights: relative size (normalized by image area) of annotation the pixel belongs to.
    """
    assert type(polygons) == list, "polygons should be a list"

    # Filter out zero-area polygons
    polygons = [polygon for polygon in polygons if 0 < polygon.area]

    # tic = time.time()

    channel_count = fill + edges + vertices
    polygons_raster = np.zeros((*shape, channel_count), dtype=np.uint8)
    distance_maps = np.ones((*shape, len(polygons)))  # Init with max value (distances are normed)
    sizes = np.ones(shape)  # Init with max value (sizes are normed)
    image_area = shape[0] * shape[1]
    for i, polygon in enumerate(polygons):
        minx, miny, maxx, maxy = polygon.bounds
        mini = max(0, math.floor(miny) - 2*line_width)
        minj = max(0, math.floor(minx) - 2*line_width)
        maxi = min(polygons_raster.shape[0], math.ceil(maxy) + 2*line_width)
        maxj = min(polygons_raster.shape[1], math.ceil(maxx) + 2*line_width)
        bbox_shape = (maxi - mini, maxj - minj)
        bbox_polygon = shapely.affinity.translate(polygon, xoff=-minj, yoff=-mini)
        bbox_raster = draw_polygons([bbox_polygon], bbox_shape, fill, edges, vertices, line_width, antialiasing)
        polygons_raster[mini:maxi, minj:maxj] = np.maximum(polygons_raster[mini:maxi, minj:maxj], bbox_raster)
        bbox_mask = 0 < np.sum(bbox_raster, axis=2)  # Polygon interior + edge + vertex
        if bbox_mask.max():  # Make sure mask is not empty
            polygon_mask = np.zeros(shape, dtype=np.bool_)
            polygon_mask[mini:maxi, minj:maxj] = bbox_mask
            polygon_dist = cv.distanceTransform(1 - polygon_mask.astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_5,
                                        dstType=cv.CV_64F)
            polygon_dist /= (polygon_mask.shape[0] + polygon_mask.shape[1])  # Normalize dist
            distance_maps[:, :, i] = polygon_dist

            selem = skimage.morphology.disk(line_width)
            bbox_dilated_mask = skimage.morphology.binary_dilation(bbox_mask, footprint=selem)
            sizes[mini:maxi, minj:maxj][bbox_dilated_mask] = polygon.area / image_area

    polygons_raster = np.clip(polygons_raster, 0, 255)
    # skimage.io.imsave("polygons_raster.png", polygons_raster)

    if edges:
        edge_channels = -1 + fill + edges
        # Remove border edges because they correspond to cut buildings:
        polygons_raster[:line_width, :, edge_channels] = 0
        polygons_raster[-line_width:, :, edge_channels] = 0
        polygons_raster[:, :line_width, edge_channels] = 0
        polygons_raster[:, -line_width:, edge_channels] = 0

    distances = compute_distances(distance_maps)
    # skimage.io.imsave("distances.png", distances)

    distances = distances.astype(np.float16)
    sizes = sizes.astype(np.float16)

    # toc = time.time()
    # print(f"Rasterize {len(polygons)} polygons: {toc - tic}s")

    return polygons_raster, distances, sizes


def compute_distances(distance_maps):
    distance_maps.sort(axis=2)
    distance_maps = distance_maps[:, :, :2]
    distances = np.sum(distance_maps, axis=2)
    return distances


def draw_polygons(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    assert type(polygons) == list, "polygons should be a list"
    assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
        polygons = [shapely.affinity.scale(polygon, xfact=2.0, yfact=2.0, origin=(0, 0)) for polygon in polygons]
        line_width *= 2
    else:
        draw_shape = shape
    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (draw_shape[1], draw_shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(polygon.exterior.coords, fill=255)
            for interior in polygon.interiors:
                draw.polygon(interior.coords, fill=0)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(polygon.exterior.coords, fill=255, width=line_width)
            for interior in polygon.interiors:
                draw.line(interior.coords, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            for vertex in polygon.exterior.coords:
                torch_lydorn.torchvision.transforms.functional.draw_circle(draw, vertex, line_width / 2, fill=255)
            for interior in polygon.interiors:
                for vertex in interior.coords:
                    torch_lydorn.torchvision.transforms.functional.draw_circle(draw, vertex, line_width / 2, fill=255)

    im_list = []
    if antialiasing:
        # resize images:
        for im_draw in im_draw_list:
            resize_shape = (shape[1], shape[0])
            im_list.append(im_draw[0].resize(resize_shape, Image.BILINEAR))
    else:
        for im_draw in im_draw_list:
            im_list.append(im_draw[0])

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im) for im in im_list]
    array = np.stack(array_list, axis=-1)
    return array


def _rasterize_coco(image, polygons):
    import pycocotools.mask as cocomask

    image_size = image.shape[:2]
    mask = np.zeros(image_size)
    for polygon in polygons:
        rle = cocomask.frPyObjects([np.array(polygon.exterior.coords).reshape(-1)], image_size[0], image_size[1])
        m = cocomask.decode(rle)

        for i in range(m.shape[-1]):
            mi = m[:, :, i]
            mi = mi.reshape(image_size)
            mask += mi
    return mask


def _test():
    import skimage.io

    rasterize = Rasterize(fill=True, edges=False, vertices=False, line_width=2, antialiasing=True, return_distances=True, return_sizes=True)

    image = np.zeros((300, 300))
    polygons = [
        shapely.geometry.Polygon([
            [10.5, 10.5],
            [100, 10],
            [100, 150],
            [10, 100],
            [10, 10],
        ]),
        shapely.geometry.Polygon([
            [10+150, 10],
            [100+150, 10],
            [100+150, 100],
            [10+150, 100],
            [10+150, 10],
        ]),
    ]
    polygons_raster, distances, size_weights = rasterize(image, polygons)

    skimage.io.imsave('rasterize.polygons_raster.png', polygons_raster)
    skimage.io.imsave('rasterize.distances.png', distances)
    skimage.io.imsave('rasterize.size_weights.png', size_weights)

    # Rasterize with pycocotools
    coco_mask = _rasterize_coco(image, polygons)
    skimage.io.imsave('rasterize.coco_mask.png', coco_mask)


if __name__ == "__main__":
    _test()