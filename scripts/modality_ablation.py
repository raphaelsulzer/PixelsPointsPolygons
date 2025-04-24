import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose

from pixelspointspolygons.predict import FFLPredictor, HiSupPredictor, Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf
from pixelspointspolygons.eval import Evaluator


def predict_all():
    
    experiments = ["lidar_density_ablation4", "lidar_density_ablation16"]

    experiments = [
        # Pix2Poly
        "p2p_image",
        "p2p_lidar", 
        "p2p_fusion",
        # HiSup 
        "hisup_image", 
        "hisup_lidar",
        "hisup_fusion"
        ]
    

    setup_hydraconf()

    res_dict = {}
    split = "val"
    
    # run_type = "debug"
    run_type = "release"

    with initialize(config_path="../config", version_base="1.3"):
        for exp in experiments:
            cfg = compose(config_name="config", 
                          overrides=[f"experiment={exp}",
                                     "checkpoint=best_val_iou",
                                     f"run_type={run_type}"])
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
            
            
            cfg.eval.pred_file = cfg.eval.pred_file.replace("predictions", f"predictions_{split}")
            predictor.predict_dataset(split="val")
    
                
            ### Evaluate
            ee = Evaluator(cfg)
            ee.load_gt(cfg.dataset.annotations[split])
            ee.load_predictions(cfg.eval.pred_file)
            res_dict[f"{cfg.experiment.model.name}/{cfg.experiment.name}"]=ee.evaluate()

        
        df = pd.DataFrame.from_dict(res_dict, orient='index')

        # pd.concat(df_list, axis=0, ignore_index=False)
        # Save the DataFrame to a CSV file
        # output_dir = os.path.join(self.cfg.host.data_root, "eval_results")
        
        print("\n")
        print(df)
        print("\n")
        
        cfg.eval.eval_file = f"{cfg.eval.eval_file}_modality_ablation.csv"
        
        print(f"Save eval file to {cfg.eval.eval_file}")
        df.to_csv(cfg.eval.eval_file, index=True, float_format="%.3g")
    
        ee.to_latex(csv_file=cfg.eval.eval_file)




# @hydra.main(config_path="../config", config_name="config", version_base="1.3")
# def main(cfg):
        
#     setup_omegaconf(cfg)
    
#     predict_all(cfg)

    
        
if __name__ == "__main__":
    predict_all()
