from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import shapely.geometry
import shapely.affinity
from scipy.ndimage.morphology import distance_transform_edt

from functools import partial

from . import functional


class AngleFieldInit(object):
    def __init__(self, line_width=1):
        self.line_width = line_width

    def __call__(self, image, polygons):
        size = (image.shape[0], image.shape[1])
        return init_angle_field(polygons, size, self.line_width)


def init_angle_field(polygons, shape, line_width=1):
    """
    Angle field {\theta_1} the tangent vector's angle for every pixel, specified on the polygon edges.
    Angle between 0 and pi.
    This is not invariant to symmetries.

    :param polygons:
    :param shape:
    :return: (angles: np.array((num_edge_pixels, ), dtype=np.uint8),
              mask: np.array((num_edge_pixels, 2), dtype=np.int))
    """
    assert type(polygons) == list, "polygons should be a list"
    if len(polygons):
        assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    im = Image.new("L", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    for polygon in polygons:
        draw_linear_ring(draw, polygon.exterior.coords, line_width)
        for interior in polygon.interiors:
            draw_linear_ring(draw, interior, line_width)

    # Convert image to numpy array
    array = np.array(im)
    return array


def draw_linear_ring(draw, linear_ring, line_width):
    # --- edges:
    coords = np.array(linear_ring)
    edge_vect_array = np.diff(coords, axis=0)
    edge_angle_array = np.angle(edge_vect_array[:, 1] + 1j * edge_vect_array[:, 0])  # ij coord sys
    neg_indices = np.where(edge_angle_array < 0)
    edge_angle_array[neg_indices] += np.pi

    first_uint8_angle = None
    for i in range(coords.shape[0] - 1):
        edge = (coords[i], coords[i + 1])
        angle = edge_angle_array[i]
        uint8_angle = int((255 * angle / np.pi).round())
        if first_uint8_angle is None:
            first_uint8_angle = uint8_angle
        line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
        draw.line(line, fill=uint8_angle, width=line_width)
        functional.draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)

    # Add first vertex back on top (equals to last vertex too):
    functional.draw_circle(draw, line[1], radius=line_width / 2, fill=first_uint8_angle)
