import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..misc import suppress_stdout

from .angle_eval import compute_max_angle_error
from .cIoU import compute_IoU_cIoU
from .polis import compute_polis
from .topdig_metrics import compute_mask_metrics



def compute_coco_metrics(annFile, resFile):
    
    print("WARNING: The area thresholds for the COCO evaluation seem to be specific for bigger images?!")
    
    type=1
    annType = ['bbox', 'segm']

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    # imgIds = cocoGt.getImgIds()
    
    # Force area to "all"

    # print("\npred: ", cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=[0]))[2])
    # print("\ngt: ", cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[0]))[1])
    
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=annType[type])
    
    # cocoEval.params.areaRng = [[0**2, 1e10**2]]  # effectively disables area filtering
    # cocoEval.params.areaRngLbl = ['all']
    
    # cocoEval.params.areaRng = [
    # [0, 1e10**2],       # "small"
    # [0, 1e10**2],      # "medium"
    # [0, 1e10**2], # "large"
    # ]
    # cocoEval.params.areaRngLbl = ['small', 'medium', 'large']
    

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


def compute_boundary_coco_metrics(annFile, resFile):
    
    print("WARNING: Boundary IoU evaluation is super slow.")
    
    try:
        from boundary_iou.coco_instance_api.coco import COCO as BCOCO
        from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
    except ImportError:
        print("Could not import boundary IoU API. Please install it by running the following commands:")
        print("git clone git@github.com:bowenc0221/boundary-iou-api.git")
        print("cd boundary_iou_api")
        print("pip install -e .")
        return {}

    dilation_ratio = 0.02 # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    return cocoEval.stats



def evaluate(gt_file, dt_file, modes=["coco"], exp_name="Experiment 1", outfile=None, num_workers=8):
    
    res_dict = {}
    
        
    with suppress_stdout():

        if "polis" in modes:  
            res_dict.update(compute_polis(gt_file, dt_file))
        if "mta" in modes:
            res_dict.update(compute_max_angle_error(gt_file, dt_file, num_workers=num_workers))
        if "iou" in modes:
            res_dict.update(compute_IoU_cIoU(dt_file, gt_file))
        if "topdig" in modes:
            res_dict.update(compute_mask_metrics(dt_file, gt_file))
        if "boundary-coco" in modes:
            res_dict.update(compute_boundary_coco_metrics(gt_file, dt_file))
        if "coco" in modes:
            res_dict.update(compute_coco_metrics(gt_file, dt_file))
    
    print(f"\nResults for {dt_file}:")
    
    
    df = pd.DataFrame.from_dict(res_dict, orient='index').transpose()
    
    # Set the row name
    df.index = [exp_name]
    
    # Format the DataFrame to display only two digits after the comma
    pd.options.display.float_format = "{:.2f}".format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    # pd.set_option('display.max_colwidth', 50)
    
    # Pretty print the DataFrame
    print(df)
    
    return res_dict    
        
