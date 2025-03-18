import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.predict import Predictor

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    pp = Predictor(cfg)
    pp.predict()

if __name__ == "__main__":
    main()
