"""Object cropping utilities - crop objects from images using bbox annotations."""
import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

from .data_loader import (
    load_yolo_annotations,
    load_json_attributes,
    load_classes,
    yolo_to_pixel_bbox,
    parse_xanylabeling_json,
    match_annotations_to_attributes,
    discover_attributes
)


def crop_single_image(
    image_path: str,
    txt_path: str,
    json_path: Optional[str] = None,
    padding: int = 5,
    min_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Crop all annotated objects from a single image.
    
    Args:
        image_path: Path to the source image
        txt_path: Path to YOLO format annotation txt
        json_path: Optional path to X-AnyLabeling JSON with attributes
        padding: Padding around bbox in pixels
        min_size: Minimum crop dimension
    
    Returns:
        List of dicts with: crop_image (PIL), class_id, bbox, attributes, label
    """
    # Robust image loading: skip unreadable/corrupt images
    try:
        img = Image.open(image_path)
        img.load()  # Force full load to catch truncated images
        if img.mode not in ('RGB', 'RGBA', 'L', 'P'):
            img = img.convert('RGB')
    except Exception as e:
        # Cannot open image - return empty list (caller should skip)
        return []
    img_w, img_h = img.size
    
    # Load YOLO annotations (already skips malformed lines)
    yolo_anns = load_yolo_annotations(txt_path)
    
    # Load JSON attributes if available (skip if corrupt)
    json_objects = []
    if json_path and os.path.exists(json_path):
        try:
            json_data = load_json_attributes(json_path)
            if 'shapes' in json_data:
                json_objects = parse_xanylabeling_json(json_data)
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            pass  # Skip bad JSON, continue with YOLO-only
    
    # Match annotations to attributes
    if json_objects:
        merged = match_annotations_to_attributes(
            yolo_anns, json_objects, img_w, img_h
        )
    else:
        merged = []
        for ann in yolo_anns:
            bbox = yolo_to_pixel_bbox(ann, img_w, img_h, padding)
            merged.append({
                'class_id': ann['class_id'],
                'bbox': bbox,
                'yolo_ann': ann,
                'attributes': {},
                'label': '',
                'match_iou': 0
            })
    
    # Crop each object
    results = []
    for idx, obj in enumerate(merged):
        x1, y1, x2, y2 = obj['bbox']
        
        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)
        
        # Check minimum size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        
        crop = img.crop((x1, y1, x2, y2))
        
        results.append({
            'crop_image': crop,
            'class_id': obj['class_id'],
            'bbox': (x1, y1, x2, y2),
            'attributes': obj['attributes'],
            'label': obj.get('label', ''),
            'source_image': os.path.basename(image_path),
            'crop_index': idx
        })
    
    return results


def crop_dataset(
    images_dir: str,
    labels_dir: str,
    json_dir: Optional[str] = None,
    output_dir: str = "./data/cropped",
    classes_file: Optional[str] = None,
    padding: int = 5,
    min_size: int = 32
) -> Dict[str, Any]:
    """
    Crop all objects from a dataset of images.
    
    Saves cropped images to output_dir organized by class.
    Returns statistics about the cropping process.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class names
    class_names = []
    if classes_file and os.path.exists(classes_file):
        class_names = load_classes(classes_file)
    
    # Find all images
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in img_extensions
    ])
    
    stats = {
        'total_images': len(image_files),
        'total_crops': 0,
        'crops_per_class': {},
        'images_processed': 0,
        'errors': []
    }
    
    for img_name in tqdm(image_files, desc="Cropping objects"):
        img_path = os.path.join(images_dir, img_name)
        stem = Path(img_name).stem
        
        # Find corresponding txt file
        txt_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(txt_path):
            stats['errors'].append(f"No txt annotation for {img_name}")
            continue
        
        # Find corresponding JSON file
        json_path = None
        if json_dir:
            json_path = os.path.join(json_dir, f"{stem}.json")
            if not os.path.exists(json_path):
                json_path = None
        
        try:
            crops = crop_single_image(
                img_path, txt_path, json_path,
                padding=padding, min_size=min_size
            )
            
            for crop_info in crops:
                class_id = crop_info['class_id']
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                # Create class directory
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save crop
                crop_filename = f"{stem}_obj{crop_info['crop_index']}.jpg"
                crop_path = os.path.join(class_dir, crop_filename)
                crop_info['crop_image'].save(crop_path, quality=95)
                
                # Save attributes metadata
                if crop_info['attributes']:
                    meta_path = os.path.join(class_dir, f"{stem}_obj{crop_info['crop_index']}.json")
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'source_image': crop_info['source_image'],
                            'class_id': class_id,
                            'class_name': class_name,
                            'label': crop_info['label'],
                            'bbox': crop_info['bbox'],
                            'attributes': crop_info['attributes']
                        }, f, ensure_ascii=False, indent=2)
                
                stats['total_crops'] += 1
                stats['crops_per_class'][class_name] = stats['crops_per_class'].get(class_name, 0) + 1
            
            stats['images_processed'] += 1
            
        except Exception as e:
            stats['errors'].append(f"Error processing {img_name}: {str(e)}")
    
    return stats


def organize_by_attribute(
    cropped_dir: str,
    json_dir: Optional[str] = None,
    master_json: Optional[str] = None,
    attribute_name: str = "material",
    output_dir: str = "./data/datasets/material",
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Organize cropped images into classification dataset folders by attribute value.
    
    For a given attribute (e.g. 'material'), creates folders:
    output_dir/
        leather/
            img1.jpg
            img2.jpg
        wood/
            img3.jpg
        ...
    
    Args:
        cropped_dir: Directory with cropped images (may have class subdirs)
        json_dir: Directory with per-image JSON attribute files
        master_json: Path to master JSON with all attributes
        attribute_name: Which attribute to organize by
        output_dir: Output dataset directory
        target_size: Resize images to this size
    
    Returns:
        Statistics dict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'attribute': attribute_name,
        'total_images': 0,
        'values': {},
        'skipped_no_attribute': 0,
        'errors': []
    }
    
    # Load master JSON if provided
    master_data = {}
    if master_json and os.path.exists(master_json):
        with open(master_json, 'r', encoding='utf-8') as f:
            master_data = json.load(f)
    
    # Walk through cropped directory
    for root, dirs, files in os.walk(cropped_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            fpath = os.path.join(root, fname)
            stem = Path(fname).stem
            
            # Try to find attribute value
            attr_value = None
            
            # Method 1: Look for companion JSON file
            meta_json = os.path.join(root, f"{stem}.json")
            if os.path.exists(meta_json):
                try:
                    with open(meta_json, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    attr_value = meta.get('attributes', {}).get(attribute_name)
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Method 2: Look in per-image JSON dir
            if attr_value is None and json_dir:
                # Extract original image name from crop filename (e.g. img001_obj0 -> img001)
                parts = stem.rsplit('_obj', 1)
                if len(parts) == 2:
                    orig_stem = parts[0]
                    obj_idx = int(parts[1]) if parts[1].isdigit() else 0
                    per_img_json = os.path.join(json_dir, f"{orig_stem}.json")
                    if os.path.exists(per_img_json):
                        try:
                            jdata = load_json_attributes(per_img_json)
                            if 'shapes' in jdata:
                                shapes = jdata['shapes']
                                if obj_idx < len(shapes):
                                    attr_value = shapes[obj_idx].get('attributes', {}).get(attribute_name)
                                    if attr_value is None:
                                        attr_value = shapes[obj_idx].get('flags', {}).get(attribute_name)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
            
            # Method 3: Look in master JSON
            if attr_value is None and master_data:
                parts = stem.rsplit('_obj', 1)
                if len(parts) == 2:
                    orig_name = parts[0]
                    obj_idx = int(parts[1]) if parts[1].isdigit() else 0
                    # Try multiple key formats
                    for key in [orig_name, f"{orig_name}.jpg", f"{orig_name}.png"]:
                        if key in master_data:
                            img_info = master_data[key]
                            objs = img_info.get('objects', [])
                            if obj_idx < len(objs):
                                attr_value = objs[obj_idx].get('attributes', {}).get(attribute_name)
                            break
            
            if attr_value is None or str(attr_value).strip() == '':
                stats['skipped_no_attribute'] += 1
                continue
            
            # Clean attribute value for directory name
            attr_value = str(attr_value).strip()
            safe_value = attr_value.replace('/', '_').replace(chr(92), '_').replace(' ', '_')
            
            # Copy and resize image to attribute value folder
            value_dir = os.path.join(output_dir, safe_value)
            os.makedirs(value_dir, exist_ok=True)
            
            try:
                img = Image.open(fpath)
                if target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                out_path = os.path.join(value_dir, fname)
                img.save(out_path, quality=95)
                
                stats['total_images'] += 1
                stats['values'][attr_value] = stats['values'].get(attr_value, 0) + 1
            except Exception as e:
                stats['errors'].append(f"Error processing {fname}: {str(e)}")
    
    return stats
