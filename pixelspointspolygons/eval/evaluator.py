import pandas as pd

from .angle_eval import compute_max_angle_error
from .cIoU import compute_IoU_cIoU
from .polis import compute_polis
from .topdig_metrics import compute_mask_metrics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_coco_metrics(annFile, resFile):
    type=1
    annType = ['bbox', 'segm']
    print('Running demo for *%s* results.' % (annType[type]))

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
    
    df = pd.DataFrame()
    
    if "coco" in modes:
        res = compute_coco_metrics(gt_file, dt_file)
    if "polis" in modes:  
        compute_polis(gt_file, dt_file)
    if "mta" in modes:
        compute_max_angle_error(gt_file, dt_file)
    if "iou" in modes:
        compute_IoU_cIoU(dt_file, gt_file)
    if "topdig" in modes:
        compute_mask_metrics(dt_file, gt_file)
        
    a=5
        
