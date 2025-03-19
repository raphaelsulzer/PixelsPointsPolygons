import logging
import os
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..misc import suppress_stdout, make_logger

from .angle_eval import compute_max_angle_error
from .cIoU import compute_IoU_cIoU
from .polis import compute_polis
from .topdig_metrics import compute_mask_metrics

class Evaluator:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        self.gt_file = cfg.eval.gt_file
        self.pred_file = cfg.eval.pred_file
        
        with suppress_stdout():
            self.cocoGt = COCO(cfg.eval.gt_file)
        # self.cocoDt = self.cocoGt.loadRes(self.eval.pred_file)
        self.cocoDt = None
        
        logging_level = getattr(logging, cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger("Evaluator",level=logging_level)

        
    def load_predictions(self, pred_file=None):
        
        if pred_file is None:
            pred_file = self.pred_file
        else:
            if not os.path.isfile(pred_file):
                raise FileExistsError(f"File {pred_file} does not exist.")
            self.pred_file = pred_file
        
        self.logger.info(f"Loading predictions from {pred_file}")
        with suppress_stdout():
            self.cocoDt = self.cocoGt.loadRes(pred_file)
        

    def compute_coco_metrics(self, annType='segm'):
        
        self.logger.warning("The area thresholds for the COCO evaluation seem to be specific for bigger images?!")

        # imgIds = cocoGt.getImgIds()
        
        # Force area to "all"

        # print("\npred: ", cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=[0]))[2])
        # print("\ngt: ", cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[0]))[1])
        
        cocoEval = COCOeval(self.cocoGt, self.cocoDt, iouType=annType)
        

        # cocoEval.params.imgIds = [0]
        cocoEval.params.catIds = [100]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Create a dictionary to store the COCO evaluation metrics
        coco_metrics = {
            'AP': cocoEval.stats[0],
            'AP50': cocoEval.stats[1],
            'AP75': cocoEval.stats[2],
            'AP_small': cocoEval.stats[3],
            'AP_medium': cocoEval.stats[4],
            'AP_large': cocoEval.stats[5],
            'AR1': cocoEval.stats[6],
            'AR10': cocoEval.stats[7],
            'AR100': cocoEval.stats[8],
            'AR_small': cocoEval.stats[9],
            'AR_medium': cocoEval.stats[10],
            'AR_large': cocoEval.stats[11]
        }
        
        return coco_metrics


    def compute_boundary_coco_metrics(self):
        
        self.logger.warning("Boundary IoU evaluation is super slow.")
        
        try:
            from boundary_iou.coco_instance_api.coco import COCO as BCOCO
            from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
        except ImportError:
            self.logger.warning("Could not import boundary IoU API. Please install it by running the following commands:")
            self.logger.warning("git clone git@github.com:bowenc0221/boundary-iou-api.git")
            self.logger.warning("cd boundary_iou_api")
            self.logger.warning("pip install -e .")
            return {}

        dilation_ratio = 0.02 # default settings 0.02
        cocoGt = BCOCO(self.gt_file, get_boundary=True, dilation_ratio=dilation_ratio)
        cocoDt = cocoGt.loadRes(self.pred_file)
        cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        return cocoEval.stats


    def print_info(self):
        
        self.logger.info(f"Dataset info for {self.cocoGt.dataset['info']}...")
        self.logger.info(f"Number of images (gt/pred): {len(self.cocoGt.getImgIds())}/{len(self.cocoDt.getImgIds())}")
        self.logger.info(f"Number of polygons (gt/pred): {len(self.cocoGt.getAnnIds())}/{len(self.cocoDt.getAnnIds())}")
        
        # num_vert_gt = 0
        # num_vert_pred = 0
        # for ann in self.cocoGt.dataset['annotations']:
        #     if len(ann['segmentation']):
        #         num_vert_gt += len(ann['segmentation'][0])//2
                
        # for ann in self.cocoDt.dataset['annotations']:
        #     if len(ann['segmentation']):
        #         num_vert_pred += len(ann['segmentation'][0])//2
 
        # self.logger.info(f"Number of vertices (gt/pred): {num_vert_gt}/{num_vert_pred}")


    def evaluate(self):
        
        if self.cocoDt is None:
            raise ValueError("No predictions loaded. Please load predictions first with the load_predictions() method.")
        
        res_dict = {}
        
        self.print_info()
                    
        with suppress_stdout():

            if "polis" in self.cfg.eval.modes:  
                res_dict.update(compute_polis(self.gt_file, self.pred_file))
            if "mta" in self.cfg.eval.modes:
                res_dict.update(compute_max_angle_error(self.gt_file, self.pred_file, num_workers=self.cfg.num_workers))
            if "iou" in self.cfg.eval.modes:
                res_dict.update(compute_IoU_cIoU(self.pred_file, self.gt_file))
            if "topdig" in self.cfg.eval.modes:
                res_dict.update(compute_mask_metrics(self.pred_file, self.gt_file))
            if "boundary-coco" in self.cfg.eval.modes:
                res_dict.update(self.compute_boundary_coco_metrics())
            if "coco" in self.cfg.eval.modes:
                res_dict.update(self.compute_coco_metrics())
        
        
        self.logger.info(f"Results for {self.pred_file}:")
        
        
        df = pd.DataFrame.from_dict(res_dict, orient='index').transpose()
        
        # Set the row name
        df.index = [self.cfg.experiment_name]
        
        # Format the DataFrame to display only two digits after the comma
        pd.options.display.float_format = "{:.2f}".format
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 500)
        # pd.set_option('display.max_colwidth', 50)
        
        # Pretty print the DataFrame
        print(df)
        
        return res_dict    
            
