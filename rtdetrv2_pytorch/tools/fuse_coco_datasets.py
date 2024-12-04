import json
import os

def filter_coco_by_class(coco_file, target_class="person"):
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    # Get the category ID for the target class
    target_cat_ids = [cat['id'] for cat in data['categories'] if cat['name'] == target_class]
    if not target_cat_ids:
        raise ValueError(f"Class '{target_class}' not found in {coco_file}.")
    
    target_cat_id = target_cat_ids[0]
    
    # Filter annotations to keep only those of the target class
    filtered_annotations = [
        ann for ann in data['annotations'] if ann['category_id'] == target_cat_id
    ]
    
    # Get image IDs corresponding to the filtered annotations
    image_ids = set(ann['image_id'] for ann in filtered_annotations)
    
    # Filter images to include only those used by the annotations
    filtered_images = [
        img for img in data['images'] if img['id'] in image_ids
    ]
    
    # Create the filtered COCO data
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': [cat for cat in data['categories'] if cat['id'] == target_cat_id],
    }
    
    return filtered_data

# Example usage
dataset1_path = "/path/to/dataset1.json"
dataset2_path = "/path/to/dataset2.json"

dataset1_filtered = filter_coco_by_class(dataset1_path)
dataset2_filtered = filter_coco_by_class(dataset2_path)

# Save filtered datasets
with open('filtered_dataset1.json', 'w') as f:
    json.dump(dataset1_filtered, f)

with open('filtered_dataset2.json', 'w') as f:
    json.dump(dataset2_filtered, f)

"""Merge into one
def merge_coco_datasets(dataset1, dataset2):
    merged = {
        'images': [],
        'annotations': [],
        'categories': dataset1['categories'],  # Assuming categories are the same
    }
    
    # Track ID mappings to avoid duplicates
    image_id_offset = max(img['id'] for img in dataset1['images']) + 1
    annotation_id_offset = max(ann['id'] for ann in dataset1['annotations']) + 1
    
    # Add dataset1
    merged['images'].extend(dataset1['images'])
    merged['annotations'].extend(dataset1['annotations'])
    
    # Add dataset2 with updated IDs
    for img in dataset2['images']:
        img['id'] += image_id_offset
        merged['images'].append(img)
    
    for ann in dataset2['annotations']:
        ann['id'] += annotation_id_offset
        ann['image_id'] += image_id_offset
        merged['annotations'].append(ann)
    
    return merged

# Load filtered datasets
with open('filtered_dataset1.json', 'r') as f:
    filtered_dataset1 = json.load(f)

with open('filtered_dataset2.json', 'r') as f:
    filtered_dataset2 = json.load(f)

# Merge datasets
merged_dataset = merge_coco_datasets(filtered_dataset1, filtered_dataset2)

# Save merged dataset
with open('merged_dataset.json', 'w') as f:
    json.dump(merged_dataset, f)



"""