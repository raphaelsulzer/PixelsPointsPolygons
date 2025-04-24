import json
from pathlib import Path
import hydra

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    """
    Loads a COCO annotation file, keeps annotations for the first 25% of images,
    and saves the result to a new file.

    Args:
        input_path (str or Path): Path to the original COCO JSON file.
        output_path (str or Path): Path to the filtered output JSON file.
    """
    input_path = cfg.dataset.annotations["test"]
    output_path = input_path.replace(".json", "_subset25.json")

    with open(input_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']

    # Get the first 25% of images
    num_images = len(images)
    num_to_keep = max(1, num_images // 4)
    kept_images = images[:num_to_keep]
    kept_image_ids = {img['id'] for img in kept_images}

    # Keep annotations that belong to the kept image IDs
    kept_annotations = [ann for ann in annotations if ann['image_id'] in kept_image_ids]

    # Prepare the filtered dataset
    filtered_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': kept_images,
        'annotations': kept_annotations,
        'categories': coco.get('categories', [])
    }

    # Save to output file
    with open(output_path, 'w') as f:
        json.dump(filtered_coco, f, indent=2)

    print(f"Filtered COCO saved to {output_path}")
    

if __name__ == "__main__":
    main()