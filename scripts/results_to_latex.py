import hydra

from pixelspointspolygons.misc.shared_utils import setup_hydraconf
from pixelspointspolygons.eval import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    

    # outfile = "/home/rsulzer/overleaf/building_footprint_dataset/tables/modality_ablation_table.tex"
        
    # TODO: check why HiSup NY failed on jz
    
    
    # TODO: make the density ablation table
    
    # TODO: merge the annotation files of the 3 countries to one annotations_all.json
    
    cfg.eval.split = "test"
    
    
    ### MODALITY ABLATION ###
    # cfg.eval.eval_file = f"{cfg.eval.eval_file}_modality_ablation_{cfg.country}_{cfg.eval.split}.csv"
    # ee = Evaluator(cfg)
    
    # caption = r"\textbf{Modality ablation}. We compare the baseline models trained and tested on different modalities. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    # ee.to_latex(csv_file=cfg.eval.eval_file, 
    #             caption=caption,
    #             label="tab:modality_ablation", type="modality")


    # ### DENSITY ABLATION ###
    # cfg.eval.eval_file = f"{cfg.eval.eval_file}_lidar_density_ablation_{cfg.country}_{cfg.eval.split}.csv"
    # ee = Evaluator(cfg)
    
    # caption = r"\textbf{Density ablation}. We compare a ViT~\cite{vit}~+~FFL~\cite{ffl} model trained and tested on increasingly dense LiDAR point clouds. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    # ee.to_latex(csv_file=cfg.eval.eval_file, 
    #             caption=caption,
    #             label="tab:density_ablation",
    #             type="density")
    
    #### ALL COUNTRIES ####
    cfg.eval.eval_file = f"{cfg.eval.eval_file}_all_countries_{cfg.country}_{cfg.eval.split}.csv"
    ee = Evaluator(cfg)
    
    caption = r"\textbf{Multimodal building polygon prediction}. We compare the baseline models with fusion encoders on the full dataset. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    ee.to_latex(csv_file=cfg.eval.eval_file, 
                caption=caption,
                label="tab:all_countries", type="all")
    
    
    
if __name__ == "__main__":
    main()
    

