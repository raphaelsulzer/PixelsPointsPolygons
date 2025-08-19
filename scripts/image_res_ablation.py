import sys

import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
from logging import getLogger

from pixelspointspolygons.predict import FFLPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf, count_trainable_parameters, parse_cli_overrides
from pixelspointspolygons.eval import Evaluator



def predict_all():
    
    logger = getLogger("HiSupPredictor rank 0")
    
    experiments = [
        # FFL
        ("ffl_image", "ffl_image_015", "224015"),
        ("ffl_image", "v4_image_bs4x16", "224"),
        ]
    
    setup_hydraconf()

    cli_overrides = parse_cli_overrides()
    
    exp_dict = {}
    with initialize(config_path="../config", version_base="1.3"):
        pbar = tqdm(total=len(experiments))
        for experiment, name, image_res in experiments:
            
            overrides = cli_overrides + \
                [f"experiment={experiment}",
                 f"experiment.name={name}",
                "checkpoint=best_val_iou",
                "eval.split=test",
                "country=Switzerland",
                f"dataset.size={image_res}"]
            cfg = compose(config_name="config", 
                          overrides=overrides)
            OmegaConf.resolve(cfg)
            
            logger.info(f"Predict {experiment}/{name} on {cfg.experiment.country}/{cfg.evaluation.split}")
            # pbar.set_description(f"Predict and evaluate {experiment} on {cfg.evaluation.split}")
            pbar.refresh()  
          
            #############################################
            ################## PREDICT ##################
            #############################################

            local_rank, world_size = setup_ddp(cfg)
                        
            predictor = FFLPredictor(cfg, local_rank, world_size)

            # time_dict = predictor.predict_dataset(split=cfg.evaluation.split)
            # res_dict.update(time_dict)
            # res_dict["num_params"] = count_trainable_parameters(predictor.model)/1e6
            
            logger.info(f"Evaluate {experiment}/{name} on {cfg.experiment.country}/{cfg.evaluation.split}")
            
                      
            #############################################
            ################## EVALUATE #################
            #############################################
            
            ### Evaluate
            ee = Evaluator(cfg)
            ee.pbar_disable = False
            ee.load_gt(cfg.dataset.annotations[cfg.evaluation.split])
            ee.load_predictions(cfg.evaluation.pred_file)
            res_dict=ee.evaluate(print_info=False)

            
            exp_dict[f"{cfg.experiment.model.name}/{cfg.experiment.name}"] = res_dict
            
            pbar.update(1)

        pbar.close()
        df = pd.DataFrame.from_dict(exp_dict, orient='index')
        
        print("\n")
        print(df)
        print("\n")
        
        cfg.evaluation.eval_file = f"{cfg.evaluation.eval_file}_image_res_ablation_{cfg.experiment.country}_{cfg.evaluation.split}.csv"
        
        logger.info(f"Save eval file to {cfg.evaluation.eval_file}")
        df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
    
        caption = r"\textbf{Ground sampling distance ablation}. We compare a ViT~\cite{vit}~+~FFL~\cite{ffl} model trained and tested on aerial images with a GSD of 15 and 25~cm. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
        ee.to_latex(csv_file=cfg.evaluation.eval_file, 
                    caption=caption,
                    label="tab:gsd_ablation",
                    type="resolution")




# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
