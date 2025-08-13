
import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')
import json
from collections import Counter
import cv2
import numpy as np

def yolov8_to_coco(image_dir, label_dir, output_file):
    """Converts YOLOv8 annotations to COCO format.

    Args:
        image_dir: Path to the directory containing images.
        label_dir: Path to the directory containing YOLOv8 label files (.txt).
        output_file: Path to the output COCO JSON file.
    """

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_id_counter = 1  # Start category IDs from 1
    annotation_id_counter = 1

    # Create category entries
    # Assuming categories are defined in your label files
    # Adapt this part based on your specific category names and IDs

    # Example: assuming categories are numbers from 0...N
    # You may need to get these from other means

    categories = set()  # Use set to store unique categories
    for label_filename in os.listdir(label_dir):
      with open(os.path.join(label_dir, label_filename), 'r') as f:
        for line in f:
          class_id = int(line.strip().split()[0])
          categories.add(class_id)


    for cat_id in sorted(list(categories)):
        coco_data["categories"].append({"id": category_id_counter, "name": str(cat_id)})
        category_id_counter += 1


    for image_filename in os.listdir(image_dir):
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files

        image_path = os.path.join(image_dir, image_filename)
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (640, 640),  interpolation=cv2.INTER_LINEAR)
        
            if img.shape[-1] != 3:
                img = np.dstack([img, img, img])
            height, width, _ = img.shape
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue #Skip images which cannot be opened

        coco_data["images"].append({
            "id": annotation_id_counter,
            "file_name": image_filename,
            "width": width,
            "height": height
        })
        
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
    
        cnt = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    
                    w = int(w)
                    h = int(h)
                    x1 = int(x_center - w / 2) 
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2) 
                    
                    coco_data["annotations"].append({
                        "id": cnt,
                        "image_id": annotation_id_counter,
                        "category_id": int(class_id) + 1, #Assuming class ids start at 0 in your labels
                        "bbox": [x1, y1, w, h],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0
                    })
                    
                    cnt += 1
                    
        annotation_id_counter += 1

    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)


sub_folder = '..' # train or valid
# Example usage
image_dir = f"./dataset/{sub_folder}/images" # Replace with your image directory
label_dir = f"./dataset/{sub_folder}/labels" # Replace with your labels directory
output_file = f"./dataset/{sub_folder}/labels.json"
yolov8_to_coco(image_dir, label_dir, output_file)


json_path = f'./dataset/{sub_folder}//corrected_labels.json'

# Load the JSON file
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Extract all annotation IDs
annotation_ids = [ann['id'] for ann in coco_data['annotations']]

# Find duplicates
duplicates = [item for item, count in Counter(annotation_ids).items() if count > 1]

if duplicates:
    print(f"Duplicate annotation IDs found: {duplicates}")
else:
    print("No duplicate annotation IDs found.")
    
# Find problematic annotations with duplicate IDs
problematic_annotations = [ann for ann in coco_data['annotations'] if ann['id'] in duplicates]

print("Problematic Annotations:")
for ann in problematic_annotations:
    print(ann)

for i, ann in enumerate(coco_data['annotations']):
    ann['id'] = i


with open('corrected_labels.json', 'w') as f:
    json.dump(coco_data, f, indent=4)