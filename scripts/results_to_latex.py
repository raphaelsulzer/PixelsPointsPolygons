import hydra

from pixelspointspolygons.misc.shared_utils import setup_hydraconf
from pixelspointspolygons.eval import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    
    cfg.eval.split = "test"
    cfg.eval.eval_file = f"{cfg.eval.eval_file}_modality_ablation_{cfg.country}_{cfg.eval.split}.csv"
    
    
    ee = Evaluator(cfg)
    # outfile = "/home/rsulzer/overleaf/building_footprint_dataset/tables/modality_ablation_table.tex"
    
    
    # TODO: check why HiSup NY failed on jz
    
    # TODO: make the density ablation table
    
    # TODO: merge the annotation files of the 3 countries to one annotations_all.json
    
    caption = r"\textbf{Modality ablation}. We compare the baseline models trained and tested on different modalities. For each metric, we highlight the \colorbox{blue!25}{best} and \colorbox{blue!10}{second best} scores."
    ee.to_latex(csv_file=cfg.eval.eval_file, 
                caption=caption,
                label="tab:modality_ablation")

if __name__ == "__main__":
    main()
    

