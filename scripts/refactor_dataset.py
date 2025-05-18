
import os
import hydra
from omegaconf import OmegaConf
from pixelspointspolygons.train import FFLTrainer, HiSupTrainer, Pix2PolyTrainer
import re

from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    steps = 5000
    a=5
    
    modality = "images"
    split = "train"
    
    country = "CH"

    for country in ["CH","NZ","NY"]:
        
        for modality in ["images","lidar"]:
            
            for split in ["train","val","test"]:
        
                path = os.path.join(cfg.dataset.path,modality,split,country)
                
                if not os.path.exists(path):
                    print(f"Path {path} does not exist")
                    continue
                
                files = os.listdir(path)
                
                for file in files:
                    
                    infile = os.path.join(path,file)
                    
                    print(infile)
                    
                    if not os.path.isfile(infile):
                        continue
                    
                    numbers = re.findall(r'\d+\.?\d*', file)
                    number = int(numbers[0])        
                    
                    outfolder = number//steps*steps
                    outfolder = os.path.join(path,f"{outfolder}")
                    os.makedirs(outfolder, exist_ok=True)
                    
                    outfile = os.path.join(outfolder,file)
                    
                    os.rename(infile, outfile)
        
    path = os.path.join(cfg.dataset.path,modality,split,country)
    
    files = os.listdir(path)
    
    for file in files:
        
        infile = os.path.join(path,file)
        
        print(infile)
        
        if not os.path.isfile(infile):
            continue
        
        numbers = re.findall(r'\d+\.?\d*', file)
        number = int(numbers[0])        
        
        outfolder = number//steps*steps
        outfolder = os.path.join(path,f"{outfolder}")
        os.makedirs(outfolder, exist_ok=True)
        
        outfile = os.path.join(outfolder,file)
        
        os.rename(infile, outfile)

        
        # if "Switzerland" in file:
        #     outfile = os.path.join(cfg.dataset.path,modality,split,"CH",file)
        # elif "NZ" in file:
        #     outfile = os.path.join(cfg.dataset.path,modality,split,"NZ",file)
        # elif "NY" in file:
        #     outfile = os.path.join(cfg.dataset.path,modality,split,"NY",file)
        # else:
        #     print(file)
        #     continue        

if __name__ == "__main__":
    main()