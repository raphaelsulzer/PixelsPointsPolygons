import logging
import os
import tqdm
import sys
import json
import re

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
from .line_dof import compute_line_dof
from .hausdorff import compute_hausdorff_chamfer
from .polis_chamfer_hausdorff import PointBasedMetrics

# Format the DataFrame to display only two digits after the comma
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

class Evaluator:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        self.gt_file = None
        self.pred_file = None
        
        self.cocoGt = None
        self.cocoDt = None
        
        self.verbosity = getattr(logging, cfg.run_type.logging.upper(), logging.INFO)
        self.logger = make_logger("Evaluator",level=self.verbosity)
        self.pbar_updata_every = cfg.host.update_pbar_every
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
    
    def load_gt(self, gt_file):
        
        # if gt_file is None:
        #     gt_file = self.gt_file
        # else:
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
        
        # self.logger.warning("The area thresholds for the COCO evaluation seem to be specific for bigger images?!")
        
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
        
        self.logger.info(f"Info for (gt/pred): {os.path.basename(self.gt_file)} / {os.path.basename(os.path.dirname(self.gt_file))}")
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


    def evaluate(self, print_info=True):
        
        if self.cocoDt is None:
            raise ValueError("No predictions loaded. Please load predictions first with the load_predictions() method.")
        
        res_dict = {}
        
        if print_info:
            self.print_info()
                    
        with suppress_stdout():
            
            
            if bool(set(self.cfg.evaluation.modes) & set(["polis", "chamfer", "hausdorff"])):  
                self.logger.info("Computing point-based metrics...")
                gt_coco = COCO(self.gt_file)
                dt_coco = gt_coco.loadRes(self.pred_file)
                polisEval = PointBasedMetrics(gt_coco, dt_coco, iou_threshold=0.5, pbar_disable=self.pbar_disable, num_workers=self.cfg.num_workers)
                res_dict.update(polisEval.evaluate())
                
                # res_dict.update(compute_polis(self.gt_file, self.pred_file, pbar_disable=self.pbar_disable))
            # if "hausdorff" in self.cfg.evaluation.modes: 
            #     self.logger.info("Computing Hausdorff and Chamfer distance...") 
            #     res_dict.update(compute_hausdorff_chamfer(self.gt_file, self.pred_file, 
            #                                               pbar_disable=self.pbar_disable,
            #                                               workers=self.cfg.num_workers))
            if "ldof" in self.cfg.evaluation.modes:
                self.logger.info("Computing line DoF...")
                if os.path.isfile(self.cfg.host.ldof_exe):
                    res_dict.update(compute_line_dof(
                        self.cfg.host.ldof_exe, self.gt_file, self.pred_file, pbar_disable=self.pbar_disable))
                else:
                    self.logger.warning(f"Line DoF executable {self.cfg.host.ldof_exe} not found. Skipping line DoF evaluation.")
            if "mta" in self.cfg.evaluation.modes:
                self.logger.info("Computing MTA...")
                res_dict.update(compute_max_angle_error(self.gt_file, self.pred_file, num_workers=self.cfg.num_workers))
            if "iou" in self.cfg.evaluation.modes:
                self.logger.info("Computing IoU and C-IoU...")
                res_dict.update(compute_IoU_cIoU(self.pred_file, self.gt_file, pbar_disable=self.pbar_disable))
            if "subset_iou" in self.cfg.evaluation.modes:
                self.logger.info("Computing Subset IoU and C-IoU...")
                res_dict.update(compute_IoU_cIoU(self.pred_file, self.gt_file, subset=True, pbar_disable=self.pbar_disable))
            if "topdig" in self.cfg.evaluation.modes:
                self.logger.info("Computing Topdig...")
                res_dict.update(compute_mask_metrics(self.pred_file, self.gt_file))
            if "boundary-coco" in self.cfg.evaluation.modes:
                self.logger.info("Computing Boundary COCO...")
                res_dict.update(self.compute_boundary_coco_metrics())
            if "coco" in self.cfg.evaluation.modes:
                self.logger.info("Computing COCO...")
                res_dict.update(self.compute_coco_metrics())
            if "stats" in self.cfg.evaluation.modes:
                self.logger.info("Computing Stats...")
                res_dict.update(self.compute_coco_stats())
        
        
        self.logger.info(f"Results for {self.pred_file}:")
        
        
        return res_dict
            
    def print_dict_results(self, res_dict):
        
        df = pd.DataFrame.from_dict(res_dict, orient='index').transpose()
        
        # Set the row name
        df.index = [self.cfg.experiment.name]
        
        # Pretty print the DataFrame
        print(df)
        
        
    def get_model_name(self, val):
        model_name = val.split('/')[0]
                
        if model_name == 'ffl':
            model_name = r"\textbf{ViT}~\cite{vit}~+~\textbf{FFL}~\cite{ffl}"
        elif model_name == 'hisup':
            model_name = r"\textbf{ViT}~\cite{vit}~+~\textbf{HiSup}~\cite{hisup}"
        elif model_name == 'pix2poly':
            model_name = r"\textbf{ViT}~\cite{vit}~+~\textbf{Pix2Poly}~\cite{pix2poly}"
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return model_name
    
    def format_model_and_modality(self, val):
        """Extract model name and modality from string like 'modelX/somelongstring'"""
        model_name = val.split('/')[0]
                
        if 'both' in val.lower() or 'fusion' in val.lower():
            cellcolor = 'magenta!10'
            # modality_str = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{Hybrid}'
            modality_str = r'\textbf{Fusion}'
            modality = "both"
        elif 'lidar' in val.lower():
            cellcolor = 'green!10'
            # modality_str = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{LiDAR}'
            modality_str = r'\textbf{LiDAR}'
            modality = "lidar"
        elif 'image' in val.lower():
            cellcolor = 'yellow!10'
            # modality_str = r'\cellcolor{'+ cellcolor + r'}' + r'\textbf{Image}'
            modality_str = r'\textbf{Image}'
            modality = "image"
        else:
            modality_str = r'\textbf{Fusion}'
            modality = "both"
            self.logger.warning(f"No modality found in experiment name. Assuming fusion.")
            # raise ValueError(f"Unknown modality name: {model_name}")
        
        if modality == "image":
            if model_name == 'ffl':
                model_name = r'\multirow{3}{*}{\shortstack{\textbf{ViT}~\cite{vit}~+ \\ \textbf{FFL}~\cite{ffl}}}'
            elif model_name == 'hisup':
                model_name = r'\multirow{3}{*}{\shortstack{\textbf{ViT}~\cite{vit}~+ \\ \textbf{HiSup}~\cite{hisup}}}'
            elif model_name == 'pix2poly':
                model_name = r'\multirow{3}{*}{\shortstack{\textbf{ViT}~\cite{vit}~+ \\ \textbf{Pix2Poly}~\cite{pix2poly}}}'
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        else:
            model_name = ""
        

        return model_name, modality_str
    
    def format_metric_name(self, names):
        
        temp = []
        for name in names:
            if name == 'IoU':
                temp.append(r'\textbf{IoU} $\uparrow$')
            elif name == 'C-IoU':
                temp.append(r'\textbf{C-IoU} $\uparrow$')
            elif name == 'POLIS':
                temp.append(r'\textbf{POLIS [m]} $\downarrow$')
            elif name == 'MTA' or name == 'mta':
                temp.append(r'\textbf{MTA [$^\circ$]} $\downarrow$')
            elif name == 'Boundary IoU':
                temp.append(r'\textbf{Boundary IoU} $\uparrow$')
            elif name == 'NR':
                temp.append(r'\textbf{NR=1}')
            elif name == "prediction_time":
                temp.append(r'\textbf{Time [s]} $\downarrow$')
            elif name == "num_params":
                temp.append(r'\textbf{Params [$\times 10^6$]} $\downarrow$')
            elif name == "AP":
                temp.append(r'\textbf{AP} $\uparrow$')
            elif name == "AR10":
                temp.append(r'\textbf{AR} $\uparrow$')
            elif name == "hausdorff":
                temp.append(r'\textbf{HD [m]} $\downarrow$')
            elif name == "chamfer":
                temp.append(r'\textbf{CD [m]} $\downarrow$')
            elif name == "norm_line_dofs":
                temp.append(r'\textbf{DoF} $\downarrow$')
            else:
                temp.append(name)
        return temp
    
    
    def get_first_and_second_best(self, col, col_vals):
        
        if col in ['IoU', 'C-IoU', 'Boundary IoU', 'NR', 'AP', 'AR10']: # higher is better
            best_val = col_vals.max()
            second__best_val = col_vals.nlargest(2).iloc[-1] if len(col_vals.unique()) > 1 else None
        elif col in ['POLIS', 'MTA', 'prediction_time', 'num_params', 'hausdorff', 'chamfer', 'norm_line_dofs']: # lower is better
            best_val = col_vals.min()
            second__best_val = col_vals.nsmallest(2).iloc[-1] if len(col_vals.unique()) > 1 else None
        else:
            best_val = None
            second__best_val = None
            # raise ValueError(f"Unknown metric: {col}")
        
        return best_val, second__best_val
    
        
    def get_metric_description(self, table_type):
        

        if table_type == "density" or table_type == "resolution":
            desc = r'    & & \multicolumn{4}{c}{\emph{Boundary}}  & \multicolumn{3}{c}{\emph{Area}}  &  \multicolumn{3}{c}{\emph{Complexity}} \\'
        elif table_type == "modality":
            desc = r'    & & & \multicolumn{4}{c}{\emph{Boundary}}& \multicolumn{3}{c}{\emph{Area}}  & \emph{Complexity} &  \multicolumn{2}{c}{\emph{Efficiency}} \\'
        elif table_type == "all":
            desc = r'    & & \multicolumn{4}{c}{\emph{Boundary}}  & \multicolumn{3}{c}{\emph{Area}}  &  \multicolumn{3}{c}{\emph{Complexity}} \\'
        else:
            raise ValueError(f"Unknown type: {table_type}")
        
        return desc
    
    def to_latex(self,df=None,csv_file=None,caption="Patch prediction",label="tab:patch",outfile=None,type="modality"):
        
        self.logger.info("Converting DataFrame to LaTeX format...")
        
        if csv_file is not None:
            df = pd.read_csv(csv_file)
        elif df is None:
            raise ValueError("Either df or csv_file must be provided.")
        else:
            raise ValueError("Either df or csv_file must be provided.")
        

        if type == "density":
            df = df.filter(items=["Unnamed: 0","POLIS", "chamfer", "hausdorff", "MTA", "AP", "AR10", "IoU", "C-IoU", "NR", "norm_line_dofs"])
        elif type == "resolution":
            df = df.filter(items=["Unnamed: 0","POLIS", "chamfer", "hausdorff", "MTA", "AP", "AR10", "IoU", "C-IoU", "NR", "norm_line_dofs"])
        elif type == "modality":
            df = df.filter(items=["Unnamed: 0","POLIS", "chamfer", "hausdorff", "MTA", "AP", "AR10", "IoU", "NR", "prediction_time", "num_params"])
        elif type == "all":
            df = df.filter(items=["Unnamed: 0","POLIS", "chamfer", "hausdorff", "MTA", "AP", "AR10", "IoU", "C-IoU", "NR", "norm_line_dofs"])
        else:
            raise ValueError(f"Unknown type: {type}")
                    
        lines = []
        lines.append(r'\begin{table}[]')


        ##### format metric #####
        cols = self.format_metric_name(df.columns)
        if type == "modality":
            cols = [r'\textbf{Model}', r'\textbf{Modality}'] + cols
            align = '@{}cc@{}'+ 'H|' + ('c' * (len(cols)-3))  + '@{}'
            lines.append(r'\setlength{\tabcolsep}{2pt}')
        elif type == "density":
            cols = [r'\textbf{Density [$pts/m^2$]}'] + cols
            align = '@{}c'+ 'H|' + ('c' * (len(cols)-2))  + '@{}'
            lines.append(r'\setlength{\tabcolsep}{2pt}')
        elif type == "resolution":
            cols = [r'\textbf{GSD [cm]}'] + cols
            align = '@{}c'+ 'H|' + ('c' * (len(cols)-2))  + '@{}'
            lines.append(r'\setlength{\tabcolsep}{2pt}')
        elif type == "all":
            cols = [r'\textbf{Model}'] + cols
            align = '@{}l'+ 'H|' + ('c' * (len(cols)-2))  + '@{}'
            lines.append(r'\setlength{\tabcolsep}{2pt}')
        else:
            raise ValueError(f"Unknown type: {type}")
        
        lines.append(r'\centering')
        lines.append(r'\resizebox{\textwidth}{!}{')
        lines.append(r'\begin{tabular}{' + align + '}')
        lines.append(r'\toprule')
        
        ### add metric description
        metric_desc = self.get_metric_description(type)
        lines.append(metric_desc)
        
        # lines.append(r'\midrule')
        lines.append(r'\midrule')
        lines.append(' & '.join(cols) + r' \\')
        lines.append(r'\midrule')

        ##### format model and modality #####
        model_name = ""
        for i, row in df.iterrows():
            if type == "modality":
                model, modality = self.format_model_and_modality(row.iloc[0])
                formatted_row = [model, modality]
            elif type == "density":
                density = row.iloc[0]
                density = re.search(r"mnv(\d+)$", density)
                density = int(density.group(1))//4
                formatted_row = [str(density)]
            elif type == "resolution":
                gsd = [15,25]
                formatted_row = [str(gsd[i])]
            elif type == "all":
                model = self.get_model_name(row.iloc[0])
                formatted_row = [model]
            else:   
                raise ValueError(f"Unknown type: {type}")

            if model_name != row.iloc[0].split('/')[0] and type == "modality" and i > 0:
                model_name = row.iloc[0].split('/')[0]
                model = ""
                lines.append(r'\midrule')
            # if i==1 or i==2:
            #     lines.append(r'\midrule')

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
        
        if outfile is None:
            outfile = self.cfg.evaluation.eval_file.replace('.csv', '.tex')
        with open(outfile, 'w') as f:
            f.write(latex_string)
        self.logger.info(f"Saved LaTeX table to {outfile}")