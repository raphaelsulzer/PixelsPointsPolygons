
import os
import re
import subprocess
import numpy as np
import json

from tqdm import tqdm
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from shapely import hausdorff_distance

import numpy as np
from scipy.spatial import cKDTree

def hausdorff_and_chamfer(set1, set2, workers=1):
    """
    Computes Hausdorff and Chamfer distances between two 2D point sets.

    Parameters:
    - set1: (N, 2) numpy array of 2D points
    - set2: (M, 2) numpy array of 2D points

    Returns:
    - hausdorff: Hausdorff distance (float)
    - chamfer: Chamfer distance (float)
    """
    tree1 = cKDTree(set1)
    tree2 = cKDTree(set2)

    # Directed distances
    dists1, _ = tree2.query(set1,workers=workers)
    dists2, _ = tree1.query(set2,workers=workers)

    # Hausdorff distance: max of minimal distances in both directions
    hausdorff = max(dists1.max(), dists2.max())

    # Chamfer distance: mean of minimal distances in both directions
    chamfer = dists1.mean() + dists2.mean()
    chamfer /= 2

    return hausdorff, chamfer

    
    



def compute_hausdorff_chamfer(gti_annotations, input_json, sampling_dist=0.1, pbar_disable=False, workers=1):
    
    workers = 1 if workers == 0 else workers
    
    
    # Ground truth annotations
    coco_gt = COCO(gti_annotations)

    imgInfo = coco_gt.loadImgs(0)[0]
    res = imgInfo["res_x"]
    max_dist = res * imgInfo["height"] * np.sqrt(2) # max distance in meters

    # load predicted annotations
    with open(input_json, 'r') as f:
        data = json.load(f)
    coco_dt = coco_gt.loadRes(data)

    ## get all gt images that have an annotation
    image_ids = coco_dt.getImgIds()
    
    bar = tqdm(image_ids, disable=pbar_disable)

    
    hausdorff = []
    chamfer = []
    for image_id in bar:
        
        gt_annotations = coco_gt.imgToAnns[image_id]
        dt_annotations = coco_dt.imgToAnns[image_id]
        if not len(gt_annotations) or not len(dt_annotations):
            continue 
        
        pred_points = []
        for ann in dt_annotations:
            
            seg = ann['segmentation'][0]
            pts = np.array(seg).reshape(-1, 2)
            pts = pts*res # convert pixels to meters
            poly = Polygon(pts)
            if not poly.is_valid:
                continue
            points = poly.segmentize(sampling_dist).exterior.coords
            pred_points.append(list(points))
        

        
        gt_points = []
        for ann in gt_annotations:
            seg = ann['segmentation'][0]
            pts = np.array(seg).reshape(-1, 2)
            pts = pts*res # convert pixels to meters
            poly = Polygon(pts)
            if not poly.is_valid:
                continue
            points = poly.segmentize(sampling_dist).exterior.coords
            gt_points.append(list(points))
        
        if not len(gt_points) and not len(pred_points):
            continue
        elif not len(gt_points) and len(pred_points):
            hausdorff.append(max_dist)
            chamfer.append(2)
            continue
        elif len(gt_points) and not len(pred_points):
            hausdorff.append(max_dist)
            chamfer.append(2)
            continue
        
        gt_points = np.concatenate(gt_points, axis=0)
        pred_points = np.concatenate(pred_points, axis=0)
        dists = hausdorff_and_chamfer(gt_points, pred_points, workers=workers)
        assert dists[0] <= max_dist, f"Hausdorff distance {dists[0]} is larger than max distance {max_dist}"
        hausdorff.append(dists[0])
        chamfer.append(dists[1])
    
    res_dict = {
        "hausdorff": np.mean(hausdorff),
        "chamfer": np.mean(chamfer)
    }
    
    return res_dict
    
            
