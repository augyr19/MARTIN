import json
import os
from pathlib import Path

# --- Configuration ---
BASE_DIR =          r"C:/Users/brand/Documents/College/2025/MARTIN/SYBIL/"
ANNOTATIONS_DIR =   BASE_DIR + r"annotations"
LABELS_OUTPUT_DIR = BASE_DIR + r"labels"
SPLITS_OUTPUT_DIR = BASE_DIR + r"splits"
IMAGES_DIR =        BASE_DIR + r"images"

# Ensure directories exist
os.makedirs(LABELS_OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLITS_OUTPUT_DIR, exist_ok=True)

def coco_to_yolo(x_min, y_min, w, h, img_w, img_h):
    """Converts COCO bbox to YOLO normalized center coordinates."""
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def process_annotations():
    # 1. Generate all .txt labels from the full master file
    master_json = os.path.join(ANNOTATIONS_DIR, "full_plastopol.json")
    print(f"Reading master annotations: {master_json}")
    
    with open(master_json, 'r') as f:
        data = json.load(f)

    # Create image lookup (id -> info)
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    img_to_ann = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_ann:
            img_to_ann[img_id] = []
        img_to_ann[img_id].append(ann)

    # Write YOLO .txt files
    print("Generating .txt label files...")
    for img_id, img_info in images.items():
        img_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        txt_filename = Path(img_name).stem + ".txt"
        txt_path = os.path.join(LABELS_OUTPUT_DIR, txt_filename)
        
        with open(txt_path, 'w') as f:
            if img_id in img_to_ann:
                for ann in img_to_ann[img_id]:
                    # Shift class index 1 -> 0
                    cls_id = ann['category_id'] - 1 
                    bbox = ann['bbox'] # [x, y, w, h]
                    
                    yolo_bbox = coco_to_yolo(bbox[0], bbox[1], bbox[2], bbox[3], img_w, img_h)
                    f.write(f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")

    # 2. Generate the .txt split files (train_fold-X, val_fold-X, etc.)
    print("Generating split files...")
    json_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json') and f != "full_plastopol.json"]
    
    for j_file in json_files:
        with open(os.path.join(ANNOTATIONS_DIR, j_file), 'r') as f:
            split_data = json.load(f)
        
        # Determine output filename (removing 'plastopol_' prefix to match your screenshot)
        split_name = j_file.replace('plastopol_', '').replace('.json', '.txt')
        split_txt_path = os.path.join(SPLITS_OUTPUT_DIR, split_name)
        
        with open(split_txt_path, 'w') as f:
            for img in split_data['images']:
                # Write the full path to the image as required by YOLO split files
                img_full_path = os.path.join(IMAGES_DIR, img['file_name'])
                f.write(f"{img_full_path}\n")
        
    print(f"✅ Success! Labels in {LABELS_OUTPUT_DIR} and Splits in {SPLITS_OUTPUT_DIR}")

if __name__ == "__main__":
    process_annotations()