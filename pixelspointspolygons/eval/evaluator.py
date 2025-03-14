import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..misc import suppress_stdout

from .angle_eval import compute_max_angle_error
from .cIoU import compute_IoU_cIoU
from .polis import compute_polis
from .topdig_metrics import compute_mask_metrics



def compute_coco_metrics(annFile, resFile):
    type=1
    annType = ['bbox', 'segm']

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [100]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats


def evaluate(gt_file, dt_file, modes=["coco"], outfile=None):
    
    res_dict = {}
    
    with suppress_stdout():
        if "coco" in modes:
            res_dict.update(compute_coco_metrics(gt_file, dt_file))
        if "polis" in modes:  
            res_dict.update(compute_polis(gt_file, dt_file))
        if "mta" in modes:
            res_dict.update(compute_max_angle_error(gt_file, dt_file))
        if "iou" in modes:
            res_dict.update(compute_IoU_cIoU(dt_file, gt_file))
        if "topdig" in modes:
            res_dict.update(compute_mask_metrics(dt_file, gt_file))
    
    
    print(f"\nResults for {dt_file}:")
    
    
    df = pd.DataFrame.from_dict(res_dict, orient='index').transpose()
    
    # Format the DataFrame to display only two digits after the comma
    pd.options.display.float_format = "{:.2f}".format
    
    # Pretty print the DataFrame
    print(df)
    
    # print(res_dict)
    
        
