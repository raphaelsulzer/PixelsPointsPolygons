import sys

import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
from logging import getLogger

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf, count_trainable_parameters
from pixelspointspolygons.eval import Evaluator

def parse_cli_overrides():
    # Skip the script name
    return [arg for arg in sys.argv[1:] if "=" in arg]

def predict_all():
    
    logger = getLogger("HiSupPredictor rank 0")
    
    experiments = [
        # FFL
        ("lidar_density_ablation4", "v5_lidar_bs2x16_mnv4"),
        ("lidar_density_ablation8", "v5_lidar_bs2x16_mnv8"),
        ("lidar_density_ablation16", "v5_lidar_bs2x16_mnv16"),
        ("lidar_density_ablation32", "v5_lidar_bs2x16_mnv32"),
        ("lidar_density_ablation64", "v5_lidar_bs2x16_mnv64"),
        ("lidar_density_ablation128", "v5_lidar_bs2x16_mnv128"),
        ]
    
    # TODO: modify the prediction output file so that it does not overwrite the modality ablation predictions
    # TODO: add Marions metric!!

    setup_hydraconf()

    cli_overrides = parse_cli_overrides()
    
    exp_dict = {}
    with initialize(config_path="../config", version_base="1.3"):
        pbar = tqdm(total=len(experiments))
        for experiment, name in experiments:
            
            overrides = cli_overrides + \
                [f"experiment={experiment}",
                 f"experiment.name={name}",
                "checkpoint=best_val_iou"]
            cfg = compose(config_name="config", 
                          overrides=overrides)
            OmegaConf.resolve(cfg)
            
            logger.info(f"Predict {experiment}/{name} on {cfg.country}/{cfg.eval.split}")
            # pbar.set_description(f"Predict and evaluate {experiment} on {cfg.eval.split}")
            pbar.refresh()  
          
            #############################################
            ################## PREDICT ##################
            #############################################

            local_rank, world_size = setup_ddp(cfg)
                        
            predictor = FFLPredictor(cfg, local_rank, world_size)

            # cfg.eval.pred_file = cfg.eval.pred_file.replace("predictions", f"predictions_{cfg.country}_{cfg.eval.split}")
            time_dict = predictor.predict_dataset(split=cfg.eval.split)

            logger.info(f"Evaluate {experiment}/{name} on {cfg.country}/{cfg.eval.split}")
            
                      
            #############################################
            ################## EVALUATE #################
            #############################################
            
            ### Evaluate
            ee = Evaluator(cfg)
            ee.load_gt(cfg.dataset.annotations[cfg.eval.split])
            ee.load_predictions(cfg.eval.pred_file)
            res_dict=ee.evaluate(print_info=False)
            res_dict.update(time_dict)

            res_dict["num_params"] = count_trainable_parameters(predictor.model)/1e6
            
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
        
        cfg.eval.eval_file = f"{cfg.eval.eval_file}_lidar_density_ablation_{cfg.country}_{cfg.eval.split}.csv"
        
        logger.info(f"Save eval file to {cfg.eval.eval_file}")
        df.to_csv(cfg.eval.eval_file, index=True, float_format="%.3g")
    
        ee.to_latex(csv_file=cfg.eval.eval_file)




# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
