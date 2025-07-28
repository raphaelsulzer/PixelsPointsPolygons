import os
import json
import glob

def clean_coco_annotations(json_path, data_dir):
    
    
    missing_lidar = 0
    missing_images = 0
    
    # Load the original JSON
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Build a set of valid image IDs
    valid_images = []
    valid_image_ids = set()

    for img in coco['images']:
        missing = False
        image_path = os.path.join(data_dir, img['image_path'])
        if not os.path.exists(image_path):
            missing = True
            missing_images += 1
            # print(f"Missing file: {image_path} — removing image.")
            
        lidar_path = os.path.join(data_dir, img['lidar_path'])
        if not os.path.exists(lidar_path):
            missing = True
            missing_lidar += 1
        
        if not missing:
            valid_images.append(img)
            valid_image_ids.add(img['id'])

    # Filter annotations to only include those linked to valid images
    valid_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in valid_image_ids]

    # Overwrite the dictionary
    coco['images'] = valid_images
    coco['annotations'] = valid_annotations
    
    print(f"Removed {missing_images} images and {missing_lidar} LiDAR files from the dataset.")

    # Save cleaned JSON
    with open(json_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Cleaned annotation file saved to {json_path}")

# Example usage:
# clean_coco_annotations("annotations.json", "images/")


if __name__ == "__main__":

    
    ann_path = "/data/rsulzer/PixelsPointsPolygons_dataset/data/224/annotations"
    
    ann_files = os.listdir(ann_path)
    
    for ann_file in ann_files:
        
        
        print(f"Processing annotation file: {ann_file}")
        json_path = os.path.join(ann_path,ann_file)

        
        clean_coco_annotations(json_path=json_path, data_dir="/data/rsulzer/PixelsPointsPolygons_dataset/data/224")