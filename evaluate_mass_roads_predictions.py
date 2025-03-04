import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
import cv2

from eval.hisup_eval_utils.metrics.cIoU import calc_IoU
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score

def calc_f1score(mask: np.ndarray, mask_gti: np.ndarray):
    mask = torch.from_numpy(mask)
    mask_gti = torch.from_numpy(mask_gti)
    return binary_f1_score(preds=mask, target=mask_gti)


def calc_acc(mask: np.ndarray, mask_gti: np.ndarray):
    mask = torch.from_numpy(mask)
    mask_gti = torch.from_numpy(mask_gti)
    return binary_accuracy(preds=mask, target=mask_gti)


def compute_mask_metrics(predictions_dir, gt_dir):
    # Ground truth annotations
    # gt_masks = os.listdir(gt_dir)

    # Predictions annotations
    pred_masks = os.listdir(predictions_dir)


    images = pred_masks
    bar = tqdm(images)

    list_acc_topo = []
    list_f1_topo = []
    list_iou_topo = []

    for image_id in bar:

        # img = cv2.imread(os.path.join(predictions_dir, image_id))

        # Predictions
        topo_mask = cv2.imread(os.path.join(predictions_dir, image_id))
        topo_mask = (topo_mask != 0).astype(np.float32)

        # Ground truth
        topo_mask_gt = cv2.imread(os.path.join(gt_dir, f"{image_id.split('.')[0]}.tif"))
        topo_mask_gt = (topo_mask_gt != 0).astype(np.float32)

        # Standard Torchmetrics Implementation
        pacc_orig = calc_acc(topo_mask, topo_mask_gt)
        list_acc_topo.append(pacc_orig)
        iou_orig = calc_IoU(topo_mask, topo_mask_gt)
        list_iou_topo.append(iou_orig)
        f1score_orig = calc_f1score(topo_mask, topo_mask_gt)
        list_f1_topo.append(f1score_orig)

        bar.set_description("iou-topo: %2.4f, p-acc-topo: %2.4f, f1-topo:%2.4f " % (np.mean(list_iou_topo), np.mean(list_acc_topo), np.mean(list_f1_topo)))
        bar.refresh()

    print("Done!")
    print("############## TOPO METRICS ############")
    print("Mean IoU-Topo: ", np.mean(list_iou_topo))
    print("Mean P-Acc-Topo: ", np.mean(list_acc_topo))
    print("Mean F1-Score-Topo: ", np.mean(list_f1_topo))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", default="")
    parser.add_argument("--dt-dir", default="")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    dt_dir = args.dt_dir
    compute_mask_metrics(predictions_dir=dt_dir,
                    gt_dir=gt_dir)
