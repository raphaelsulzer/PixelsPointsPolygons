import hydra
from omegaconf import OmegaConf
from pixelspointspolygons import Trainer

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    tt = Trainer(cfg)
    tt.train()

if __name__ == "__main__":
    main()