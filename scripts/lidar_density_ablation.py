import sys

import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
from logging import getLogger

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf, parse_cli_overrides
from pixelspointspolygons.eval import Evaluator



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
        ("lidar_density_ablation256", "v5_lidar_bs2x16_mnv256"),
        ("lidar_density_ablation512", "v5_lidar_bs2x16_mnv512"),
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
                "checkpoint=best_val_iou",
                "eval.split=test",
                "country=Switzerland"]
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

        # pd.concat(df_list, axis=0, ignore_index=False)
        # Save the DataFrame to a CSV file
        # output_dir = os.path.join(self.cfg.host.data_root, "eval_results")
        
        print("\n")
        print(df)
        print("\n")
        
        cfg.evaluation.eval_file = f"{cfg.evaluation.eval_file}_lidar_density_ablation_{cfg.experiment.country}_{cfg.evaluation.split}.csv"
        
        logger.info(f"Save eval file to {cfg.evaluation.eval_file}")
        df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
    
        ee.to_latex(csv_file=cfg.evaluation.eval_file)




# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
