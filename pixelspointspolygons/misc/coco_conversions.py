import torch
import numpy as np
from skimage.measure import label, regionprops
from pycocotools import mask as coco_mask

def poly_to_bbox(poly):
    """
    input: poly----2D array with points
    """

    lt_x, lt_y = np.min(poly, axis=0)
    w = np.max(poly[:,0]) - lt_x
    h = np.max(poly[:,1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]

def generate_coco_ann(polygon_list, img_id):
    sample_ann = []
    for polygon in polygon_list:
        if polygon.shape[0] < 3:
            continue
        if isinstance(polygon, torch.Tensor):
            polygon = polygon.cpu().numpy()
        ann_per_building = {
                'image_id': int(img_id),
                'category_id': 100,
                'segmentation': [polygon.ravel().tolist()],
                'bbox': poly_to_bbox(polygon),
                # 'score': float(scores[i]),
            }
        sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_mask(mask, img_id):
    sample_ann = []
    props = regionprops(label(mask > 0.50))
    for prop in props:
        if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
            prop_mask = np.zeros_like(mask, dtype=np.uint8)
            prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

            masked_instance = np.ma.masked_array(mask, mask=(prop_mask != 1))
            score = masked_instance.mean()
            encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
            ann_per_building = {
                'image_id': img_id,
                'category_id': 100,
                'segmentation': {
                    "size": encoded_region["size"],
                    "counts": encoded_region["counts"].decode()
                },
                'score': float(score),
            }
            sample_ann.append(ann_per_building)

    return sample_ann
