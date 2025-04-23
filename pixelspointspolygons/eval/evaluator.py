import logging
import os
import tqdm
import sys
import json

import pandas as pd
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from copy import deepcopy
from tqdm import tqdm

from ..misc import suppress_stdout, make_logger, get_experiment_type

from .angle_eval import compute_max_angle_error
from .cIoU import compute_IoU_cIoU
from .polis import compute_polis
from .topdig_metrics import compute_mask_metrics

# Format the DataFrame to display only two digits after the comma
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

class Evaluator:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        self.gt_file = cfg.eval.gt_file
        self.pred_file = cfg.eval.pred_file
        
        self.cocoGt = None
        self.cocoDt = None
        
        self.verbosity = getattr(logging, cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger("Evaluator",level=self.verbosity)
        self.pbar_updata_every = cfg.update_pbar_every
        self.pbar_disable = self.verbosity >= logging.INFO


    def progress_bar(self,item):
        
        
        pbar = tqdm(item, total=len(item), 
                      file=sys.stdout, 
                    #   dynamic_ncols=True, 
                      mininterval=self.pbar_update_every,                      
                      disable=self.pbar_disable,
                      position=0,
                      leave=True)
    
        return pbar
    
    def load_gt(self, gt_file=None):
        
        if gt_file is None:
            gt_file = self.gt_file
        else:
            if not os.path.isfile(gt_file):
                raise FileExistsError(f"File {gt_file} does not exist.")
            self.gt_file = gt_file
        
        self.logger.info(f"Loading ground truth from {gt_file}")
        with suppress_stdout():
            self.cocoGt = COCO(self.gt_file)
    
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
        
        # have to pass a deepcopy here, otherwise the self.cocoGt object will be modified
        cocoEval = COCOeval(deepcopy(self.cocoGt), deepcopy(self.cocoDt), iouType=annType)
        

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


    def compute_coco_stats(self):
        
        
        # Load the COCO ground truth and prediction files
        with open(self.gt_file, 'r') as f:
            gt_data = json.load(f)
        gt_annotations = gt_data["annotations"]
        gt_num_images = len(gt_data["images"])
        
        with open(self.pred_file, 'r') as f:
            pred_annotations = json.load(f)

        # Helper function to count polygons and vertices in a COCO dataset
        def count_polygons_and_vertices(annotations):
            num_polygons = 0
            num_vertices = 0
            num_images = set()
            num_images_with_polygons = set()
            
            for annotation in tqdm(annotations,desc="Compute stats"):
                if 'segmentation' in annotation:
                    if isinstance(annotation['segmentation'], list):
                        num_polygons += 1
                        num_vertices += sum(len(seg) // 2 for seg in annotation['segmentation'])
                        num_images_with_polygons.add(annotation['image_id'])
                num_images.add(annotation['image_id'])
            return num_polygons, num_vertices, len(num_images_with_polygons)

        # Extracting ground truth and prediction data
        gt_num_polygons, gt_num_vertices, gt_num_images_with_polygons = count_polygons_and_vertices(gt_annotations)
        pred_num_polygons, pred_num_vertices, pred_num_images_with_polygons = count_polygons_and_vertices(pred_annotations)


        # Prepare the result dictionary
        result = {
            "#images": gt_num_images,
            "GT #images w/ polys": gt_num_images_with_polygons,
            "Pred #images w/ polys": pred_num_images_with_polygons,
            # "Pred #images": pred_num_images,
            "GT #polygons": gt_num_polygons,
            "Pred #polygons": pred_num_polygons,
            "GT #vertices": gt_num_vertices,
            "Pred #vertices": pred_num_vertices
        }

        return result        
        
        

    def print_info(self):
        
        self.logger.info(f"Dataset info for {self.cocoGt.dataset['info']}...")
        self.logger.info(f"Number of images (gt/pred): {len(self.cocoGt.getImgIds())}/{len(self.cocoDt.getImgIds())}")
        self.logger.info(f"Number of polygons (gt/pred): {len(self.cocoGt.getAnnIds())}/{len(self.cocoDt.getAnnIds())}")
        
        num_vert_gt = 0
        num_vert_pred = 0
        for ann in self.cocoGt.dataset['annotations']:
            if len(ann['segmentation']):
                num_vert_gt += len(ann['segmentation'][0])//2
                
        for ann in self.cocoDt.dataset['annotations']:
            if len(ann['segmentation']):
                num_vert_pred += len(ann['segmentation'][0])//2
 
        self.logger.info(f"Number of vertices (gt/pred): {num_vert_gt}/{num_vert_pred}")


    def evaluate(self):
        
        if self.cocoDt is None:
            raise ValueError("No predictions loaded. Please load predictions first with the load_predictions() method.")
        
        res_dict = {}
        
        self.print_info()
                    
        with suppress_stdout():

            if "polis" in self.cfg.eval.modes:  
                res_dict.update(compute_polis(self.gt_file, self.pred_file, pbar_disable=self.pbar_disable))
            if "mta" in self.cfg.eval.modes:
                res_dict.update(compute_max_angle_error(self.gt_file, self.pred_file, num_workers=self.cfg.num_workers))
            if "iou" in self.cfg.eval.modes:
                res_dict.update(compute_IoU_cIoU(self.pred_file, self.gt_file, pbar_disable=self.pbar_disable))
            if "topdig" in self.cfg.eval.modes:
                res_dict.update(compute_mask_metrics(self.pred_file, self.gt_file))
            if "boundary-coco" in self.cfg.eval.modes:
                res_dict.update(self.compute_boundary_coco_metrics())
            if "coco" in self.cfg.eval.modes:
                res_dict.update(self.compute_coco_metrics())
            if "stats" in self.cfg.eval.modes:
                res_dict.update(self.compute_coco_stats())
        
        
        self.logger.info(f"Results for {self.pred_file}:")
        
        
        return res_dict
            
    def print_dict_results(self, res_dict):
        
        df = pd.DataFrame.from_dict(res_dict, orient='index').transpose()
        
        # Set the row name
        df.index = [self.cfg.experiment_name]
        
        # Pretty print the DataFrame
        print(df)
        
        
    def check_if_predictions_exist(self):
        
        for item in self.cfg.experiments:
            
            for exp in item.experiment_name:
                
                name, img_dim, polygonization_method = get_experiment_type(exp)
                                                
                pred = self.cfg.checkpoint
                pred_file = os.path.join(self.cfg.host.data_root,
                                         f"{item.model}_outputs",self.cfg.dataset.name,
                                         img_dim,name,
                                         "predictions",polygonization_method,f"{pred}.json")
                if not os.path.isfile(pred_file):
                    raise FileExistsError(f"{pred_file} does not exist!")

        
        return True
    
    def evaluate_all(self):
        
        self.logger.info("Evaluating all models...")
        
        # first quickly check if the prediction file exists before doing the more lengthy evaluation
        self.check_if_predictions_exist()
        self.logger.debug("All prediction files exist.")
        
        res_dict = {}
        
        for item in self.cfg.experiments:
            
            for exp in item.experiment_name:
                
                name, img_dim, polygonization_method = get_experiment_type(exp)

                pred = self.cfg.checkpoint

                self.logger.info(f"Evaluate {item.model}/{exp}/{pred}")
                
                pred_file = os.path.join(self.cfg.host.data_root,
                                         f"{item.model}_outputs",self.cfg.dataset.name,
                                         img_dim,name,
                                         "predictions",polygonization_method,f"{pred}.json")                
                if not os.path.isfile(pred_file):
                    raise FileExistsError(f"{pred_file} does not exist!")
                
                gt_file = os.path.join(self.cfg.host.data_root,self.cfg.dataset.name,img_dim,"annotations_val.json")
                if not os.path.isfile(gt_file):
                    raise FileExistsError(f"{gt_file} does not exist!")
                
                self.load_gt(gt_file)
                self.load_predictions(pred_file)
                
                res_dict[f"{item.model}/{exp}"]=self.evaluate()

        
        df = pd.DataFrame.from_dict(res_dict, orient='index')

        # pd.concat(df_list, axis=0, ignore_index=False)
        # Save the DataFrame to a CSV file
        # output_dir = os.path.join(self.cfg.host.data_root, "eval_results")
        
        print("\n")
        print(df)
        print("\n")
        
        self.logger.info(f"Save eval file to {self.cfg.eval.eval_file}")
        df.to_csv(self.cfg.eval.eval_file, index=True, float_format="%.3g")
        
    
    def format_model_and_modality(self, val):
        """Extract model name and modality from string like 'modelX/somelongstring'"""
        model_name = val.split('/')[0]
        
        
        if 'both' in val.lower() or 'fusion' in val.lower():
            cellcolor = 'magenta!10'
            modality = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{Both}'
        elif 'lidar' in val.lower():
            cellcolor = 'green!10'
            modality = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{LiDAR}'
        elif 'image' in val.lower():
            cellcolor = 'yellow!10'
            modality = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{Image}'
        else:
            raise ValueError(f"Unknown modality name: {model_name}")
        
        if model_name == 'ffl':
            model_name = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{FFL} \cite{ffl}'
        elif model_name == 'hisup':
            model_name = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{HiSup} \cite{hisup}'
        elif model_name == 'pix2poly':
            model_name = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{Pix2Poly} \cite{pix2poly}'
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        

        return model_name, modality
    
    def format_metric_name(self, names):
        
        temp = []
        for name in names:
            if name == 'IoU':
                temp.append(r'\textbf{IoU} $\uparrow$')
            elif name == 'C-IoU':
                temp.append(r'\textbf{C-IoU} $\uparrow$')
            elif name == 'POLIS':
                temp.append(r'\textbf{POLIS} $\downarrow$')
            elif name == 'MTA' or name == 'mta':
                temp.append(r'\textbf{MTA [$^\circ$]} $\downarrow$')
            elif name == 'Boundary IoU':
                temp.append(r'\textbf{Boundary IoU} $\uparrow$')
            elif name == 'NR':
                temp.append(r'\textbf{NR=1}')
            else:
                temp.append(name)
        return temp
    
    
    def get_first_and_second_best(self, col, col_vals):
        
        if col in ['IoU', 'C-IoU', 'Boundary IoU', 'NR']: # higher is better
            best_val = col_vals.max()
            second__best_val = col_vals.nlargest(2).iloc[-1] if len(col_vals.unique()) > 1 else None
        elif col in ['POLIS', 'MTA']: # lower is better
            best_val = col_vals.min()
            second__best_val = col_vals.nsmallest(2).iloc[-1] if len(col_vals.unique()) > 1 else None
        else:
            best_val = None
            second__best_val = None
            # raise ValueError(f"Unknown metric: {col}")
        
        return best_val, second__best_val
    
    
    def to_latex(self,df=None,csv_file=None,caption="Patch prediction",label="tab:patch"):
        
        caption = r"\textbf{Quantitative results of patch prediction on our dataset}. We compare the baseline models trained and tested on different modalities. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
        
        self.logger.info("Converting DataFrame to LaTeX format...")
        
        if csv_file is not None:
            df = pd.read_csv(csv_file)
        elif df is None:
            raise ValueError("Either df or csv_file must be provided.")
        else:
            raise ValueError("Either df or csv_file must be provided.")

        lines = []
        lines.append(r'\begin{table}[H]')
        lines.append(r'\centering')

        # Build header: 2 extra columns for model + modality
        cols = self.format_metric_name(df.columns)
        cols = [r'\textbf{Method}', r'\textbf{Modality}'] + cols
        align = 'll'+ 'H' + ('c' * (len(cols)-1))
        lines.append(r'\resizebox{\textwidth}{!}{')
        lines.append(r'\begin{tabular}{' + align + '}')
        lines.append(r'\toprule')
        lines.append(' & '.join(cols) + r' \\')

        model_name = ""
        for _, row in df.iterrows():
            model, modality = self.format_model_and_modality(row.iloc[0])
            formatted_row = [model, modality]

            if model_name != row.iloc[0].split('/')[0]:
                model_name = row.iloc[0].split('/')[0]
                lines.append(r'\midrule')

            for col in df.columns:
                val = row[col]
                if isinstance(val, (int, float, np.number)):
                    try:
                        col_vals = pd.to_numeric(df[col], errors='coerce')
                        is_numeric_col = col_vals.notna().all()
                    except:
                        is_numeric_col = False

                    if is_numeric_col:
                        best_val, second_best_val = self.get_first_and_second_best(col,col_vals)

                        if abs(val) >= 100:
                            val_str = f'{int(val)}'  # Use fixed-point notation for other values
                        else:
                            val_str = f'{val:.3g}'  # Use scientific notation for very large or small numbers

                        if best_val is not None and val == best_val:
                            val_str = r'\cellcolor{blue!25} ' + val_str
                        elif second_best_val is not None and val == second_best_val:
                            val_str = r'\cellcolor{blue!10} ' + val_str
                    else:
                        val_str = str(val)
                else:
                    val_str = r'\detokenize{' + str(val) + '}'

                formatted_row.append(val_str)

            lines.append(' & '.join(formatted_row) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'}')
        if caption:
            lines.append(r'\caption{' + caption + '}')
        if label:
            lines.append(r'\label{' + label + '}')
        lines.append(r'\end{table}')
        
        latex_string = '\n'.join(lines)
        
        outfile = self.cfg.eval.eval_file.replace('.csv', '.tex')
        with open(outfile, 'w') as f:
            f.write(latex_string)
        self.logger.info(f"Saved LaTeX table to {outfile}")