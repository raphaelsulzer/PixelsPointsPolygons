import hydra

import pandas as pd

from pixelspointspolygons.misc.shared_utils import setup_hydraconf
from pixelspointspolygons.eval import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    

    # outfile = "/home/rsulzer/overleaf/building_footprint_dataset/tables/modality_ablation_table.tex"
    
            
    cfg.eval.split = "test"
    
    
    ### MODALITY ABLATION ###
    # cfg.eval.eval_file = f"{cfg.eval.eval_file}_modality_ablation_{cfg.country}_{cfg.eval.split}.csv"
    
    ## load time from time table
    # time_table = cfg.eval.eval_file.replace("metrics", "time")
    # df1 = pd.read_csv(cfg.eval.eval_file)
    # df2 = pd.read_csv(time_table)
    # df2 = df2.filter(items=["prediction_time", "num_params"])
    # df_combined = pd.concat([df1, df2], axis=1)
    # df_combined.to_csv(cfg.eval.eval_file, index=False)
    
    # ee = Evaluator(cfg)
    # caption = r"\textbf{Modality ablation}. We compare the baseline models trained and tested on different modalities. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    # ee.to_latex(csv_file=cfg.eval.eval_file, 
    #             caption=caption,
    #             label="tab:modality_ablation", type="modality")

    
    ### GSD ABLATION ###
    # cfg.eval.eval_file = f"{cfg.eval.eval_file}_image_res_ablation_{cfg.country}_{cfg.eval.split}.csv"

    # ee = Evaluator(cfg)
    
    # caption = r"\textbf{Ground sampling distance ablation}. We compare a ViT~\cite{vit}~+~FFL~\cite{ffl} model trained and tested on aerial images with a GSD of 15 and 25~cm. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    # ee.to_latex(csv_file=cfg.eval.eval_file, 
    #             caption=caption,
    #             label="tab:gsd_ablation",
    #             type="resolution")
    
    
    
    
    
    # ### DENSITY ABLATION ###
    cfg.eval.eval_file = f"{cfg.eval.eval_file}_lidar_density_ablation_{cfg.country}_{cfg.eval.split}.csv"

    ee = Evaluator(cfg)
    
    caption = r"\textbf{Density ablation}. We compare a ViT~\cite{vit}~+~FFL~\cite{ffl} model trained and tested on increasingly dense LiDAR point clouds. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    ee.to_latex(csv_file=cfg.eval.eval_file, 
                caption=caption,
                label="tab:density_ablation",
                type="density")
    
    #### ALL COUNTRIES ####
    # cfg.eval.eval_file = f"{cfg.eval.eval_file}_all_countries_{cfg.country}_{cfg.eval.split}.csv"
    # ee = Evaluator(cfg)
    
    # caption = r"\textbf{Multimodal building polygon prediction}. We compare the baseline models with fusion encoders on the full dataset. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    # ee.to_latex(csv_file=cfg.eval.eval_file, 
    #             caption=caption,
    #             label="tab:all_countries", type="all")
    
    
    
if __name__ == "__main__":
    main()
    

