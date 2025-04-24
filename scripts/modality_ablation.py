import sys

import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf, count_trainable_parameters
from pixelspointspolygons.eval import Evaluator

def parse_cli_overrides():
    # Skip the script name
    return [arg for arg in sys.argv[1:] if "=" in arg]

def predict_all():
    
    experiments = ["lidar_density_ablation4", "lidar_density_ablation16"]

    experiments = [
        # Pix2Poly
        "p2p_image",
        "p2p_lidar", 
        "p2p_fusion",
        # # HiSup 
        "hisup_image", 
        "hisup_lidar",
        "hisup_fusion"
        ]
    

    setup_hydraconf()

    cli_overrides = parse_cli_overrides()
    
    exp_dict = {}
    with initialize(config_path="../config", version_base="1.3"):
        for exp in experiments:
            
            overrides = cli_overrides + [f"experiment={exp}", "checkpoint=best_val_iou"]
            cfg = compose(config_name="config", 
                          overrides=overrides)
            OmegaConf.resolve(cfg)
            print(f"Running: {cfg.experiment.name}")            
            local_rank, world_size = setup_ddp(cfg)


                        
            if cfg.experiment.model.name == "ffl":
                predictor = FFLPredictor(cfg, local_rank, world_size)
            elif cfg.experiment.model.name == "hisup":
                predictor = HiSupPredictor(cfg, local_rank, world_size)
            elif cfg.experiment.model.name == "pix2poly":
                predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
            else:
                raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
            
            
            
            
            cfg.eval.pred_file = cfg.eval.pred_file.replace("predictions", f"predictions_{cfg.eval.split}")
            time_dict = predictor.predict_dataset(split=cfg.eval.split)

                
            ### Evaluate
            ee = Evaluator(cfg)
            ee.load_gt(cfg.dataset.annotations[cfg.eval.split])
            ee.load_predictions(cfg.eval.pred_file)
            res_dict=ee.evaluate()
            res_dict.update(time_dict)

            res_dict["num_params"] = count_trainable_parameters(predictor.model)/1e6
            
            exp_dict[f"{cfg.experiment.model.name}/{cfg.experiment.name}"] = res_dict

        
        df = pd.DataFrame.from_dict(exp_dict, orient='index')

        # pd.concat(df_list, axis=0, ignore_index=False)
        # Save the DataFrame to a CSV file
        # output_dir = os.path.join(self.cfg.host.data_root, "eval_results")
        
        print("\n")
        print(df)
        print("\n")
        
        cfg.eval.eval_file = f"{cfg.eval.eval_file}_modality_ablation_{cfg.eval.split}.csv"
        
        print(f"Save eval file to {cfg.eval.eval_file}")
        df.to_csv(cfg.eval.eval_file, index=True, float_format="%.3g")
    
        ee.to_latex(csv_file=cfg.eval.eval_file)




# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
