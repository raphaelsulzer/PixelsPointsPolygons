"""
The code for C-IoU, adopted from https://github.com/zorzi-s/PolyWorldPretrainedNetwork/blob/main/coco_IoU_cIoU.py.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
DATE: 2024-10-11
Description: The code is modified to handle cases where images have no annotation labels.
"""

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm

from .utils import get_pixel_mask_from_coco_seg

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        # if there is not annotation in both gt and dt, then the IoU should be 1.0
        return 1.0
    else:
        return iou

def compute_IoU_cIoU(input_json, gti_annotations, pbar_disable=False):
    # Ground truth annotations
    coco_gt = COCO(gti_annotations)

    # load predicted annotations
    with open(input_json, 'r') as f:
        data = json.load(f)
    coco_dt = coco_gt.loadRes(data)

    ## get all gt images that have an annotation
    # image_ids = coco_gt.getImgIds(catIds=coco_gt.getCatIds())
    
    ## get all dt images that have an annotation
    # image_ids = coco_dt.getImgIds(catIds=coco_dt.getCatIds())
    
    # get all gt images regardless if they have an annotation or not
    image_ids = coco_gt.getImgIds()
    
    bar = tqdm(image_ids, disable=pbar_disable)

    list_iou = []
    list_ciou = []
    list_nr = []
    for image_id in bar:
        
        mask_gt, N_GT = get_pixel_mask_from_coco_seg(coco_gt, image_id, return_n_verts=True)
        mask_dt, N_DT = get_pixel_mask_from_coco_seg(coco_dt, image_id, return_n_verts=True)

        nr = 1 - np.abs(N_DT - N_GT) / (N_DT + N_GT + 1e-9)
        iou = calc_IoU(mask_dt, mask_gt)
        list_iou.append(iou)
        list_ciou.append(iou * nr)
        list_nr.append(nr)

        bar.set_description("IoU: %2.4f, C-IoU: %2.4f, NR: %2.4f" % (np.mean(list_iou), np.mean(list_ciou), np.mean(list_nr)))
        bar.refresh()

    iou = np.mean(list_iou).item()
    ciou = np.mean(list_ciou).item()
    nr = np.mean(list_nr).item()
    
    print("IoU: %2.4f, C-IoU: %2.4f, NR: %2.4f" % (iou, ciou, nr))

    return {"IoU":iou, "C-IoU":ciou, "NR": nr}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_IoU_cIoU(input_json=dt_file,
                    gti_annotations=gt_file)
