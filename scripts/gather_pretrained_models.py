import os
import shutil

from glob import glob
from distutils.dir_util import copy_tree

experiments = [
    # Modality ablation
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
    # # GSD ablation
    # ("ffl_image", "ffl_image_015", "224015"),
    # ("ffl_image", "v4_image_bs4x16", "224"),
    # Lidar density ablation
    ("lidar_density_ablation4", "v5_lidar_bs2x16_mnv4"),
    ("lidar_density_ablation8", "v5_lidar_bs2x16_mnv8"),
    ("lidar_density_ablation16", "v5_lidar_bs2x16_mnv16"),
    ("lidar_density_ablation32", "v5_lidar_bs2x16_mnv32"),
    ("lidar_density_ablation64", "v5_lidar_bs2x16_mnv64"),
    ("lidar_density_ablation128", "v5_lidar_bs2x16_mnv128"),
    ("lidar_density_ablation256", "v5_lidar_bs2x16_mnv256"),
    ("lidar_density_ablation512", "v5_lidar_bs2x16_mnv512"),
    # all countries
    # FFL
    ("ffl_fusion", "v0_all_bs4x16"),
    # # HiSup 
    ("hisup_fusion", "v0_all_bs4x16"),
    # Pix2Poly
    ("p2p_fusion", "v0_all_bs4x16")
]

def copy():

    for model, experiment in experiments:
        
        if "ffl" in model or "lidar_density_ablation" in model:
            model = "ffl"
        elif "hisup" in model:
            model = "hisup"
        elif "p2p" in model:
            model = "pix2poly"
        else:
            raise ValueError(f"Unknown model name: {model}")

        
        infolder = f"/data/rsulzer/{model}_outputs/lidarpoly/224/{experiment}"
        outfolder = f"/data/rsulzer/PixelsPointsPolygons_results/{model}/224/{experiment}"
        
        os.makedirs(f"/data/rsulzer/PixelsPointsPolygons_results/{model}/224", exist_ok=True)
        
        print(f"Copying {infolder} to {outfolder}")
        # Copy the folder
        
        copy_tree(
            infolder,outfolder
        )

def copy_latest():

    for model, experiment in experiments:
        
        if "ffl" in model or "lidar_density_ablation" in model:
            model = "ffl"
        elif "hisup" in model:
            model = "hisup"
        elif "p2p" in model:
            model = "pix2poly"
        else:
            raise ValueError(f"Unknown model name: {model}")

        
        infile = f"/data/rsulzer/{model}_outputs/lidarpoly/224/{experiment}/checkpoints/latest.pth"
        outfile = f"/data/rsulzer/PixelsPointsPolygons_output/{model}/224/{experiment}/checkpoints/latest.pth"
        
        
        if not os.path.isfile(infile):
            print(f"File {infile} does not exist")
            continue
        
        print(f"Copying {infile} to {outfile}")
        # Copy the folder
        
        shutil.copyfile(infile, outfile)

def clean():

    for model, experiment in experiments:
        
        if "ffl" in model or "lidar_density_ablation" in model:
            model = "ffl"
        elif "hisup" in model:
            model = "hisup"
        elif "p2p" in model:
            model = "pix2poly"
        else:
            raise ValueError(f"Unknown model name: {model}")

        
        outfolder = f"/data/rsulzer/PixelsPointsPolygons_results/{model}/224/{experiment}"
        
        folders = glob(outfolder+"/predictions*")
        
        for folder in folders:
            folder = os.path.join(outfolder, folder)
            
            files = os.listdir(folder) 

            found = 0
            for file in files:
                
                if "epoch" in file:
                    print(f"Deleting {file}")
                    os.remove(os.path.join(folder, file))
                
                if "best" in file:
                    found = 1
                
                
            if not found:
                raise ValueError(f"Best file not found in {folder}")

if __name__ == "__main__":
    
    # copy()
    copy_latest()
    # clean()
    