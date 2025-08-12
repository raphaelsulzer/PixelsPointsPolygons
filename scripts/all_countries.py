import sys

import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
from logging import getLogger

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf
from pixelspointspolygons.eval import Evaluator

def parse_cli_overrides():
    # Skip the script name
    return [arg for arg in sys.argv[1:] if "=" in arg]

def predict_and_evaluate():
    
    logger = getLogger("HiSupPredictor rank 0")
    
    experiments = [
        # FFL
        ("ffl_fusion", "v0_all_bs4x16"),
        # # HiSup 
        ("hisup_fusion", "v0_all_bs4x16"),
        # Pix2Poly
        ("p2p_fusion", "v0_all_bs4x16")
        ]
    
    
    setup_hydraconf()

    cli_overrides = parse_cli_overrides()
    
    exp_dict = {}
    with initialize(config_path="../config", version_base="1.3"):
        pbar = tqdm(total=len(experiments))
        for experiment, name in experiments:
            
            overrides = cli_overrides + \
                [f"experiment={experiment}",
                 f"experiment.name={name}",
                 "country=all",
                 "eval.split=test",
                "checkpoint=best_val_iou"]
            cfg = compose(config_name="config", 
                          overrides=overrides)
            OmegaConf.resolve(cfg)
            
            logger.info(f"Predict {experiment}/{name} on {cfg.experiment.country}/{cfg.evaluation.split}")
            pbar.refresh()  
          
            #############################################
            ################## PREDICT ##################
            #############################################

            local_rank, world_size = setup_ddp(cfg)
                        
            if cfg.experiment.model.name == "ffl":
                predictor = FFLPredictor(cfg, local_rank, world_size)
            elif cfg.experiment.model.name == "hisup":
                predictor = HiSupPredictor(cfg, local_rank, world_size)
            elif cfg.experiment.model.name == "pix2poly":
                predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
            else:
                raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
            
            # time_dict = predictor.predict_dataset(split=cfg.evaluation.split)
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
            # res_dict.update(time_dict)

            exp_dict[f"{cfg.experiment.model.name}/{cfg.experiment.name}"] = res_dict
            
            pbar.update(1)

        pbar.close()
        df = pd.DataFrame.from_dict(exp_dict, orient='index')

        # pd.concat(df_list, axis=0, ignore_index=False)
        # Save the DataFrame to a CSV file
        # output_dir = os.path.join(self.cfg.host.data_root, "eval_results")
        
        print("\n")
        print(df)
        print("\n")
        
        cfg.evaluation.eval_file = f"{cfg.evaluation.eval_file}_all_countries_{cfg.experiment.country}_{cfg.evaluation.split}.csv"
        
        logger.info(f"Save eval file to {cfg.evaluation.eval_file}")
        df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
    
        ee.to_latex(csv_file=cfg.evaluation.eval_file)

    
        
if __name__ == "__main__":
    predict_and_evaluate()
