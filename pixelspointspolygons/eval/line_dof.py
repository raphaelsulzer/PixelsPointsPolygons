
import random
import string
import os
import re
import subprocess
import sys

import numpy as np
import json

from pycocotools.coco import COCO
from tqdm import tqdm


def random_string(length=5):
    chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    return ''.join(random.choices(chars, k=length))



def compute_line_dof(ldof_exe, gti_annotations, input_json, pbar_disable=False):
    
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
    
    # get only images with a pred annotation
    image_ids = coco_dt.getImgIds(catIds=coco_dt.getCatIds())

    bar = tqdm(image_ids, disable=pbar_disable)

    
    line_dofs = []
    line_segs = []
    norm_line_dofs = []
    for image_id in bar:
        
        annotations = coco_dt.imgToAnns[image_id]
        
        # if len(annotations) == 0:
        #     continue
        
        lines_per_image = []
        for ann in annotations:
            
            seg = ann['segmentation'][0]
            
            pts = np.array(seg).reshape(-1, 2)
            
            for i in range(pts.shape[0]-1):
                lines_per_image.append([pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]])
            
            
        # Convert to numpy array
        lines_per_image = np.array(lines_per_image)

        # Save to file
        output_file = random_string(4) + "_lines_image.txt"
        output_file = os.path.abspath(output_file)
        np.savetxt(output_file, lines_per_image, fmt="%.6f", delimiter=" ")
        result = subprocess.run(
            [ldof_exe, "--input", output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        os.remove(output_file) if os.path.exists(output_file) else None

        # Extract the last number from the output
        match = re.search(r"Number of degree of freedom is\s*:\s*([\d.]+)", result.stdout)
        if match:
            metric_dof = float(match.group(1))
            line_dofs.append(metric_dof)
        else:
            raise ValueError(f"{result.stdout} \n Number of degree of freedom not in result.")
        match = re.search(r"Number segments is\s*:\s*([\d.]+)", result.stdout)
        if match:
            metric_dof = float(match.group(1))
            line_segs.append(metric_dof)
        else:
            raise ValueError(f"{result.stdout} \n Number segments not in result.")
        match = re.search(r"Metric for DoF\s*:\s*([\d.]+)", result.stdout)
        if match:
            metric_dof = float(match.group(1))
            norm_line_dofs.append(metric_dof)
        else:
            raise ValueError(f"{result.stdout} \n Metric for DoF not in result.")
        
        
    
    assert len(line_dofs) == len(image_ids), "Number of line dofs does not match number of images"
    assert len(line_segs) == len(image_ids), "Number of line segments does not match number of images"
    assert len(norm_line_dofs) == len(image_ids), "Number of normalized line dofs does not match number of images"
    
    res_dict = {
        "line_dofs": np.mean(line_dofs),
        "line_segs": np.mean(line_segs),
        "norm_line_dofs": np.mean(norm_line_dofs)/100
    }
    
    return res_dict
        
            
