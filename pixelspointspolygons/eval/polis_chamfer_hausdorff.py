"""
The code is adopted from https://github.com/spgriffin/polis
"""

import numpy as np

from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as maskUtils
from shapely import geometry
from shapely.geometry import Polygon, MultiPolygon
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import multiprocessing


def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we traverse the collection of points only once, 
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]

def compute_polis(poly_a, poly_b):
    """Compares two polygons via the "polis" distance metric.
    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.
    Input:
        poly_a: A Shapely polygon.
        poly_b: Another Shapely polygon.
    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    dist = polis_scipy_dist(bndry_a.coords, bndry_b)
    dist += polis_scipy_dist(bndry_b.coords, bndry_a)
    return dist


def polis_scipy_dist(coords, bndry):
    """Computes one side of the "polis" metric.
    Input:
        coords: A Shapley coordinate sequence (presumably the vertices
                of a polygon).
        bndry: A Shapely linestring (presumably the boundary of
        another polygon).
    
    Returns:
        The "polis" metric for this pair.  You usually compute this in
        both directions to preserve symmetry.
    """
    sum = 0.0
    for pt in (geometry.Point(c) for c in coords[:-1]): # Skip the last point (same as first)
        sum += bndry.distance(pt)
    return sum/float(2*len(coords))



def get_segmentized_coords(geom, sampling_dist):
    """Segmentize a Polygon or MultiPolygon and return list of points"""
    seg = geom.segmentize(sampling_dist)
    if isinstance(seg, Polygon):
        return list(seg.exterior.coords)
    elif isinstance(seg, MultiPolygon):
        coords = []
        for p in seg.geoms:
            coords.extend(p.exterior.coords)
        return coords
    else:
        raise ValueError(f"Unexpected geometry type: {type(seg)}")


def compute_chamfer_hausdorff(poly1, poly2, sampling_dist=0.1):
    

    return chamfer_hausdorff_cdist(get_segmentized_coords(poly1, sampling_dist=sampling_dist),
                                   get_segmentized_coords(poly2, sampling_dist=sampling_dist))



def chamfer_hausdorff_cdist(set1, set2):

    set1 = np.array(set1)
    set2 = np.array(set2)
    
    if set1.ndim != 2 or set1.shape[1] != 2:
        print(f"Invalid shape for set1: {set1.shape}")
        return 0,0
    if set2.ndim != 2 or set2.shape[1] != 2:
        print(f"Invalid shape for set2: {set2.shape}")
        return 0,0
    
    # Compute full pairwise Euclidean distances
    dists = cdist(set1, set2, metric='euclidean')  # shape: (len(set1), len(set2))

    # Directed distances
    dists1 = np.min(dists, axis=1)  # from set1 to set2
    dists2 = np.min(dists, axis=0)  # from set2 to set1

    # Chamfer distance: mean of minimal distances in both directions
    chamfer = (dists1.mean() + dists2.mean()) / 2
    
    # Hausdorff distance: max of minimal distances in both directions
    hausdorff = max(dists1.max(), dists2.max())

    return chamfer, hausdorff


class PointBasedMetrics():

    def __init__(self, cocoGt=None, cocoDt=None, iou_threshold=0.5, pbar_disable=False, num_workers=0):
        self.cocoGt   = cocoGt
        self.cocoDt   = cocoDt
        self.evalImgs = defaultdict(list)
        self.eval     = {}
        self._gts     = defaultdict(list)
        self._dts     = defaultdict(list)
        self.stats    = []
        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))
        self.iou_threshold = iou_threshold
        
        self.pbar_disable = pbar_disable
        self.num_workers = num_workers  

    def _prepare(self):
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluateImg(self, imgId):
        gts = self._gts[imgId]
        dts = self._dts[imgId]

        if len(gts) == 0 or len(dts) == 0:
            return None

        gt_bboxs = [bounding_box(np.array(gt['segmentation'][0]).reshape(-1,2)) for gt in gts]
        dt_bboxs = [bounding_box(np.array(dt['segmentation'][0]).reshape(-1,2)) for dt in dts]
        gt_polygons = [np.array(gt['segmentation'][0]).reshape(-1,2) for gt in gts]
        dt_polygons = [np.array(dt['segmentation'][0]).reshape(-1,2) for dt in dts]

        # IoU match
        iscrowd = [0] * len(gt_bboxs)
        # ious = maskUtils.iou(gt_bboxs, dt_bboxs, iscrowd)
        ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

        # compute polis
        img_cd_avg = 0
        img_hd_avg = 0
        img_polis_avg = 0
        num_sample = 0
        for i, gt_poly in enumerate(gt_polygons):
            matched_idx = np.argmax(ious[:,i])
            iou = ious[matched_idx, i]
            if iou > self.iou_threshold: # iouThres:
                polis = compute_polis(Polygon(gt_poly), Polygon(dt_polygons[matched_idx]))
                chamfer, hausdorff = compute_chamfer_hausdorff(Polygon(gt_poly), Polygon(dt_polygons[matched_idx]))
                img_cd_avg += chamfer
                img_hd_avg += hausdorff
                img_polis_avg += polis
                num_sample += 1

        if num_sample == 0:
            return None
        else:
            return {"POLIS":img_polis_avg / num_sample, "chamfer": img_cd_avg / num_sample, "hausdorff": img_hd_avg / num_sample}


    def evaluate_single_process(self):
        self._prepare()
        
        res_dict = {"POLIS":0, "chamfer":0, "hausdorff":0}

        num_valid_imgs = 0
        for imgId in tqdm(self.imgIds, disable=self.pbar_disable,desc="Compute Point-based Metrics (n_workers=1)"):
            img_res = self.evaluateImg(imgId)

            if img_res is None:
                continue
            else:
                for key, val in img_res.items():
                    res_dict[key] += val
                num_valid_imgs += 1
        
        if num_valid_imgs > 0:
            for key, val in res_dict.items():
                res_dict[key] /= num_valid_imgs
                print(f'average {key}: {val}')
        else:
            for key in res_dict.keys():
                res_dict[key] = np.nan
            print('no valid images in polis evaluation')

        return res_dict

    def _evaluate_img_wrapper(args):
        self, imgId = args
        return self.evaluateImg(imgId)

    
    def evaluate(self):
        
        if self.num_workers < 2:
            return self.evaluate_single_process()
        else:
            return self.evaluate_multiprocessing(num_workers=self.num_workers)
    
    def evaluate_multiprocessing(self, num_workers=None):
        self._prepare()

        res_dict = {"POLIS": 0, "chamfer": 0, "hausdorff": 0}
        num_valid_imgs = 0

        pool = multiprocessing.Pool(
            processes=num_workers or multiprocessing.cpu_count(),
            initializer=_init_globals,
            initargs=(self._gts, self._dts, self.iou_threshold)
        )

        results = list(tqdm(pool.imap(_evaluate_single_img, self.imgIds),
                            total=len(self.imgIds),
                            disable=self.pbar_disable,
                            desc=f"Compute Point-based Metrics (n_workers={num_workers})"))

        pool.close()
        pool.join()

        for img_res in results:
            if img_res is None:
                continue
            for key, val in img_res.items():
                res_dict[key] += val
            num_valid_imgs += 1

        if num_valid_imgs > 0:
            for key in res_dict:
                res_dict[key] /= num_valid_imgs
                print(f'average {key}: {res_dict[key]}')
        else:
            for key in res_dict:
                res_dict[key] = np.nan
            print('no valid images in polis evaluation')

        return res_dict




# Global variables for each process
_global_gts = None
_global_dts = None
_global_iou_threshold = None

def _init_globals(gts, dts, iou_threshold):
    global _global_gts, _global_dts, _global_iou_threshold
    _global_gts = gts
    _global_dts = dts
    _global_iou_threshold = iou_threshold

def _evaluate_single_img(imgId):
    from shapely.geometry import Polygon
    from pycocotools import mask as maskUtils

    gts = _global_gts[imgId]
    dts = _global_dts[imgId]

    if len(gts) == 0 or len(dts) == 0:
        return None

    gt_bboxs = [bounding_box(np.array(gt['segmentation'][0]).reshape(-1,2)) for gt in gts]
    dt_bboxs = [bounding_box(np.array(dt['segmentation'][0]).reshape(-1,2)) for dt in dts]
    gt_polygons = [np.array(gt['segmentation'][0]).reshape(-1,2) for gt in gts]
    dt_polygons = [np.array(dt['segmentation'][0]).reshape(-1,2) for dt in dts]

    iscrowd = [0] * len(gt_bboxs)
    ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

    img_cd_avg = 0
    img_hd_avg = 0
    img_polis_avg = 0
    num_sample = 0
    for i, gt_poly in enumerate(gt_polygons):
        matched_idx = np.argmax(ious[:,i])
        iou = ious[matched_idx, i]
        if iou > _global_iou_threshold:
            polis = compute_polis(Polygon(gt_poly), Polygon(dt_polygons[matched_idx]))
            chamfer, hausdorff = compute_chamfer_hausdorff(Polygon(gt_poly), Polygon(dt_polygons[matched_idx]))
            img_cd_avg += chamfer
            img_hd_avg += hausdorff
            img_polis_avg += polis
            num_sample += 1

    if num_sample == 0:
        return None
    else:
        return {
            "POLIS": img_polis_avg / num_sample,
            "chamfer": img_cd_avg / num_sample,
            "hausdorff": img_hd_avg / num_sample
        }







# def compute_point_based_metrics(annFile, resFile, pbar_disable=False):
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
#     polisEval = PointBasedMetrics(gt_coco, dt_coco, iou_threshold=0.5, pbar_disable=pbar_disable)
#     return polisEval.evaluate()
    