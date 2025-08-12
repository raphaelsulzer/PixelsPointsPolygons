import json
import torch
import hydra
import pathlib

import numpy as np

from collections import defaultdict

from pixelspointspolygons.misc.shared_utils import setup_hydraconf

from ffl import FFLPreprocessing, get_offline_transform_patch

def merge_coco_annotations(input_files, output_file):
    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_set = None

    for file_path in input_files:
        with open(file_path, 'r') as f:
            print("Loading", file_path)
            data = json.load(f)

        if category_set is None:
            merged["categories"] = data["categories"]
            category_set = {cat['id'] for cat in data["categories"]}
        else:
            if {cat['id'] for cat in data["categories"]} != category_set:
                raise ValueError(f"Category IDs in {file_path} don't match the others.")

        # Map old image IDs to new ones
        image_id_map = {}
        for img in data["images"]:
            new_image_id = img["id"] + image_id_offset
            image_id_map[img["id"]] = new_image_id
            img["id"] = new_image_id
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann["id"] += annotation_id_offset
            ann["image_id"] = image_id_map[ann["image_id"]]
            merged["annotations"].append(ann)

        image_id_offset = max(img["id"] for img in merged["images"]) + 1
        annotation_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1

    
    print(f"Number of images: {len(merged['images'])}")
    print(f"Save to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

def merge_pt_files(input_files, output_file):
    
    
    data_dicts = defaultdict(list) 
    
    for input_file in input_files:
        
        data = torch.load(input_file)
        
        for k,v in data.items():
            data_dicts[k].append(v[:3])


    for k, v in data_dicts.items():
        data_dicts[k] = np.mean(v, axis=0)
    
    
    torch.save(data_dicts, output_file)
    

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):

    setup_hydraconf(cfg)
    
    regions = ["train", "val", "test"]

    for region in regions:
        
        fflp = FFLPreprocessing(cfg,
                        pre_transform=get_offline_transform_patch(),
                        fold=region)

        file = fflp.processed_flag_filepath.replace(cfg.experiment.country, "all")        
        pathlib.Path(file).touch()   
        
        
        input_files = [
            fflp.stats_filepath.replace(cfg.experiment.country, "CH"),
            fflp.stats_filepath.replace(cfg.experiment.country, "NZ"),
            fflp.stats_filepath.replace(cfg.experiment.country, "NY"),
        ]
        output_file = fflp.stats_filepath.replace(cfg.experiment.country, "all")
        merge_pt_files(input_files, output_file)
        
             
        # Example usage
        input_files = [
            fflp.ann_ffl_file.replace(cfg.experiment.country, "CH"),
            fflp.ann_ffl_file.replace(cfg.experiment.country, "NZ"),
            fflp.ann_ffl_file.replace(cfg.experiment.country, "NY"),
        ]
        output_file = fflp.ann_ffl_file.replace(cfg.experiment.country, "all")
        merge_coco_annotations(input_files, output_file)
        
        
        # Example usage
        input_files = [
            fflp.ann_file.replace(cfg.experiment.country, "CH"),
            fflp.ann_file.replace(cfg.experiment.country, "NZ"),
            fflp.ann_file.replace(cfg.experiment.country, "NY"),
        ]
        output_file = fflp.ann_file.replace(cfg.experiment.country, "all")
        merge_coco_annotations(input_files, output_file)



if __name__ == '__main__':
    main()
