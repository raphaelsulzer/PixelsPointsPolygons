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
        ("ffl_image", "v4_image_bs4x16"),
        ("ffl_lidar", "v5_lidar_bs2x16_mnv64"),
        ("ffl_fusion", "v4_fusion_bs4x16_mnv64"),
        # # HiSup 
        ("hisup_image", "v3_image_vit_cnn_bs4x12"),
        ("hisup_lidar", "lidar_pp_vit_cnn_bs2x16_mnv64"),
        ("hisup_fusion", "early_fusion_vit_cnn_bs2x16_mnv64"),
        # Pix2Poly
        ("p2p_image", "v4_image_vit_bs4x16"),
        ("p2p_lidar", "lidar_pp_vit_bs2x16_mnv64"),
        ("p2p_fusion", "early_fusion_bs2x16_mnv64"),
        ]
    
    setup_hydraconf()

    cli_overrides = parse_cli_overrides()
    
    exp_dict = {}
    with initialize(config_path="../config", version_base="1.3"):
        pbar = tqdm(total=len(experiments))
        for experiment, name in experiments:
            
            overrides = cli_overrides + \
                [f"experiment={experiment}",
                 f"experiment.name={name}", f"country=Switzerland", f"eval.split=test",
                "checkpoint=best_val_iou"]
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
            # res_dict.update(time_dict)
            # time_dict_file = f"{cfg.evaluation.eval_file}_modality_ablation_{cfg.experiment.country}_{cfg.evaluation.split}.csv".replace("metrics", "time")
            # df = pd.read_csv(time_dict_file)
            # time_dict = df.to_dict(orient="records")[0]
            

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
        
        cfg.evaluation.eval_file = f"{cfg.evaluation.eval_file}_modality_ablation_{cfg.experiment.country}_{cfg.evaluation.split}.csv"
        
        logger.info(f"Save eval file to {cfg.evaluation.eval_file}")
        df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
    
        ee.to_latex(csv_file=cfg.evaluation.eval_file)




# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
