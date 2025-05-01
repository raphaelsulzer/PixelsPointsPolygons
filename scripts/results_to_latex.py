import hydra

from pixelspointspolygons.misc.shared_utils import setup_hydraconf
from pixelspointspolygons.eval import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    setup_hydraconf(cfg)
    
    cfg.eval.eval_file = f"{cfg.eval.eval_file}_modality_ablation_{cfg.country}_{cfg.eval.split}.csv"
    ee = Evaluator(cfg)
    # outfile = "/home/rsulzer/overleaf/building_footprint_dataset/tables/modality_ablation_table.tex"
    
    # TODO: resort so that the modalities are grouped and not the models
    # add AP and AR10
    
    # TODO: check why HiSup NY failed on jz
    
    # TODO: make the density ablation table
    
    # TODO: merge the annotation files of the 3 countries to one annotations_all.json
    
    ee.to_latex(csv_file=cfg.eval.eval_file)

if __name__ == "__main__":
    main()
    

