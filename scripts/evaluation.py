# Borrowed from https://github.com/SarahwXU/HiSup/blob/main/tools/evaluation.py

import argparse

from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from eval.hisup_eval_utils.metrics.polis import PolisEval
from eval.hisup_eval_utils.metrics.angle_eval import ContourEval
from eval.hisup_eval_utils.metrics.cIoU import compute_IoU_cIoU
from eval.topdig_eval_utils.metrics.topdig_metrics import compute_mask_metrics


def coco_eval(annFile, resFile):
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

def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polisEval.evaluate()

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    parser.add_argument("--eval-type", default="coco_iou", choices=["coco_iou", "polis", "angle", "ciou", "topdig"])
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    if eval_type == 'coco_iou':
        coco_eval(gt_file, dt_file)
    elif eval_type == 'polis':
        polis_eval(gt_file, dt_file)
    elif eval_type == 'angle':
        max_angle_error_eval(gt_file, dt_file)
    elif eval_type == 'ciou':
        compute_IoU_cIoU(dt_file, gt_file)
    elif eval_type == 'topdig':
        compute_mask_metrics(dt_file, gt_file)
    else:
        raise RuntimeError('please choose a correct type from \
                            ["coco_iou", "polis", "angle", "ciou", "topdig"]')
