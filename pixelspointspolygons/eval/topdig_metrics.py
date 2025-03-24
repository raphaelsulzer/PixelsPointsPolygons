from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm

import torch
import cv2

from .cIoU import calc_IoU
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score


def calc_f1score(mask: np.ndarray, mask_gti: np.ndarray):
    union = np.logical_or(mask, mask_gti)
    U = np.sum(union)
    is_void = U == 0
    mask = torch.from_numpy(mask)
    mask_gti = torch.from_numpy(mask_gti)

    if is_void:
        return 1.0
    else:
        return binary_f1_score(preds=mask, target=mask_gti)


def calc_acc(mask: np.ndarray, mask_gti: np.ndarray):
    union = np.logical_or(mask, mask_gti)
    U = np.sum(union)
    is_void = U == 0
    mask = torch.from_numpy(mask)
    mask_gti = torch.from_numpy(mask_gti)

    if is_void:
        return 1.0
    else:
        return binary_accuracy(preds=mask, target=mask_gti)


def compute_mask_metrics(input_json, gti_annotations):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)


    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    bar = tqdm(image_ids)

    buffer_thickness = 5  # dilation factor same as that used in TopDiG.

    list_acc = []
    list_f1 = []
    list_iou = []

    list_acc_topo = []
    list_f1_topo = []
    list_iou_topo = []

    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        # Predictions
        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        topo_mask = np.zeros((img['height'], img['width']))
        poly_lines = []
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])                
            m = cocomask.decode(rle)
            if _idx == 0:
                mask = m.reshape((img['height'], img['width']))
            else:
                mask = mask + m.reshape((img['height'], img['width']))
            for ann in annotation['segmentation']:
                ann = np.array(ann).reshape(-1, 2)
                ann = np.round(ann).astype(np.int32)
                poly_lines.append(ann)
        cv2.polylines(topo_mask, poly_lines, isClosed=True, color=1., thickness=buffer_thickness)

        mask = mask != 0
        topo_mask = (topo_mask != 0).astype(np.float32)


        # Ground truth
        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        topo_mask_gt = np.zeros((img['height'], img['width']))
        poly_lines_gt = []
        for _idx, annotation in enumerate(annotations):
            if any(annotation['segmentation']):
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)
            else:
                annotation['segmentation'] = [[]]
                m = np.zeros((img['height'], img['width']))
            if m.ndim > 2:
                m = np.clip(0, 1, m.sum(axis=-1))
            if _idx == 0:
                mask_gti = m.reshape((img['height'], img['width']))
            else:
                mask_gti = mask_gti + m.reshape((img['height'], img['width']))
            for ann in annotation['segmentation']:
                ann = np.array(ann).reshape(-1, 2)
                ann = np.round(ann).astype(np.int32)
                poly_lines_gt.append(ann)
        cv2.polylines(topo_mask_gt, poly_lines_gt, isClosed=True, color=1., thickness=buffer_thickness)

        mask_gti = mask_gti != 0
        topo_mask_gt = (topo_mask_gt != 0).astype(np.float32)


        pacc = calc_acc(mask, mask_gti)
        list_acc.append(pacc)
        f1score = calc_f1score(mask, mask_gti)
        list_f1.append(f1score)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)

        pacc_topo = calc_acc(topo_mask, topo_mask_gt)
        list_acc_topo.append(pacc_topo)
        f1score_topo = calc_f1score(topo_mask, topo_mask_gt)
        list_f1_topo.append(f1score_topo)
        iou_topo = calc_IoU(topo_mask, topo_mask_gt)
        list_iou_topo.append(iou_topo)

        bar.set_description("iou: %2.4f, p-acc: %2.4f, f1:%2.4f, iou-topo: %2.4f, p-acc-topo: %2.4f, f1-topo:%2.4f " % (np.mean(list_iou), np.mean(list_acc), np.mean(list_f1), np.mean(list_iou_topo), np.mean(list_acc_topo), np.mean(list_f1_topo)))
        bar.refresh()

    
    list_iou = np.mean(list_iou).item()
    list_acc = np.mean(list_acc).item()
    list_f1 = np.mean(list_f1).item()
    list_iou_topo = np.mean(list_iou_topo).item()
    list_acc_topo = np.mean(list_acc_topo).item()
    list_f1_topo = np.mean(list_f1_topo).item()
    
    print("Mean IoU: ", list_iou)
    print("Mean P-Acc: ", list_acc)
    print("Mean F1-Score: ", list_f1)
    print("Mean IoU-Topo: ", list_iou_topo)
    print("Mean P-Acc-Topo: ", list_acc_topo)
    print("Mean F1-Score-Topo: ", list_f1_topo)
    
    return {"IoU_":list_iou, "P-Acc":list_acc, "F1-Score":list_f1, "IoU-Topo":list_iou_topo, "P-Acc-Topo":list_acc_topo, "F1-Score-Topo":list_f1_topo}
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_mask_metrics(input_json=dt_file,
                    gti_annotations=gt_file)
