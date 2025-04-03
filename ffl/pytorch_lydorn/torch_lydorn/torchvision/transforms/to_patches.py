from tqdm import tqdm

import torch

from lydorn_utils import image_utils
from lydorn_utils import polygon_utils


class ToPatches(object):
    """Splits sample into patches"""

    def __init__(self, stride, size):
        self.stride = stride
        self.size = size

    def _to_patch(self, sample):
        image = sample["image"]
        gt_polygons = sample["gt_polygons"]

        patch_boundingboxes = image_utils.compute_patch_boundingboxes(image.shape[0:2],
                                                                      stride=self.stride,
                                                                      patch_res=self.size)

        patches = []
        for patch_boundingbox in tqdm(patch_boundingboxes, desc="Patching", leave=False):
            # Crop image
            patch_image = image[patch_boundingbox[0]:patch_boundingbox[2], patch_boundingbox[1]:patch_boundingbox[3], :]
            patch_gt_polygon = polygon_utils.crop_polygons_to_patch_if_touch(gt_polygons, patch_boundingbox)
            if len(patch_gt_polygon) == 0:
                patch_gt_polygon = None
            sample["image"] = patch_image
            sample["gt_polygons"] = patch_gt_polygon
            sample["patch_bbox"] = torch.tensor(patch_boundingbox)
            patches.append(sample.copy())
        return patches

    def __call__(self, data_list):
        patch_list = []
        for data in data_list:
            patches = self._to_patch(data)
            patch_list.extend(patches)
        return patch_list
