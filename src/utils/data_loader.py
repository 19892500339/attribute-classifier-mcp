"""Data loading utilities for YOLO txt annotations and X-AnyLabeling JSON attributes."""
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_classes(classes_file: str) -> List[str]:
    """Load class names from classes.txt (one name per line)."""
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']:
        try:
            with open(classes_file, 'r', encoding=enc) as f:
                return [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            continue
    # Final fallback: read as binary
    with open(classes_file, 'rb') as f:
        return [line.decode('utf-8', errors='replace').strip() for line in f if line.strip()]


def load_yolo_annotations(txt_path: str) -> List[Dict[str, Any]]:
    """
    Load YOLO format annotation file.
    
    Each line: class_id center_x center_y width height
    All values normalized to [0, 1]
    
    Returns list of dicts with keys: class_id, cx, cy, w, h
    """
    annotations = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                annotations.append({
                    'class_id': int(parts[0]),
                    'cx': float(parts[1]),
                    'cy': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
    return annotations


def yolo_to_pixel_bbox(
    ann: Dict[str, Any],
    img_w: int,
    img_h: int,
    padding: int = 0
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized bbox to pixel coordinates.
    
    Returns (x1, y1, x2, y2) in pixels.
    """
    cx = ann['cx'] * img_w
    cy = ann['cy'] * img_h
    w = ann['w'] * img_w
    h = ann['h'] * img_h
    
    x1 = max(0, int(cx - w / 2) - padding)
    y1 = max(0, int(cy - h / 2) - padding)
    x2 = min(img_w, int(cx + w / 2) + padding)
    y2 = min(img_h, int(cy + h / 2) + padding)
    
    return (x1, y1, x2, y2)


def load_json_attributes(json_path: str) -> Dict[str, Any]:
    """
    Load a JSON attribute file (X-AnyLabeling format or custom format).
    
    X-AnyLabeling format:
    {
        "version": "...",
        "flags": {},
        "shapes": [
            {
                "label": "chair",
                "points": [[x1,y1], [x2,y2]],
                "shape_type": "rectangle",
                "flags": {"material": "leather", "has_armrest": true},
                "attributes": {"material": "leather", "has_armrest": "yes"}
            }
        ],
        "imagePath": "...",
        "imageHeight": 1080,
        "imageWidth": 1920
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_xanylabeling_json(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse X-AnyLabeling JSON format and extract objects with attributes.
    
    Returns list of dicts with: label, bbox (x1,y1,x2,y2), attributes, shape_type
    """
    objects = []
    for shape in json_data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'rectangle')
        
        # Extract bbox from points
        if shape_type == 'rectangle' and len(points) >= 2:
            x1 = min(p[0] for p in points)
            y1 = min(p[1] for p in points)
            x2 = max(p[0] for p in points)
            y2 = max(p[1] for p in points)
        elif points:
            x1 = min(p[0] for p in points)
            y1 = min(p[1] for p in points)
            x2 = max(p[0] for p in points)
            y2 = max(p[1] for p in points)
        else:
            continue
        
        # Extract attributes from both 'flags' and 'attributes' fields
        attributes = {}
        for flag_key, flag_val in shape.get('flags', {}).items():
            if isinstance(flag_val, bool):
                attributes[flag_key] = 'yes' if flag_val else 'no'
            else:
                attributes[flag_key] = str(flag_val)
        
        # 'attributes' field overrides 'flags'
        for attr_key, attr_val in shape.get('attributes', {}).items():
            attributes[attr_key] = str(attr_val) if attr_val is not None else ''
        
        # Also check for description or group_id based attributes
        desc = shape.get('description', '')
        if desc:
            # Try to parse description as key=value pairs
            for pair in desc.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    attributes[k.strip()] = v.strip()
        
        objects.append({
            'label': label,
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'attributes': attributes,
            'shape_type': shape_type
        })
    
    return objects


def match_annotations_to_attributes(
    yolo_anns: List[Dict[str, Any]],
    json_objects: List[Dict[str, Any]],
    img_w: int,
    img_h: int
) -> List[Dict[str, Any]]:
    """
    Match YOLO bbox annotations with JSON attribute annotations by IoU.
    
    Returns merged list with both bbox and attributes.
    """
    merged = []
    used_json = set()
    
    for yolo_ann in yolo_anns:
        yolo_bbox = yolo_to_pixel_bbox(yolo_ann, img_w, img_h)
        
        best_iou = 0
        best_json_idx = -1
        
        for j, json_obj in enumerate(json_objects):
            if j in used_json:
                continue
            json_bbox = json_obj['bbox']
            iou = compute_iou(yolo_bbox, json_bbox)
            if iou > best_iou:
                best_iou = iou
                best_json_idx = j
        
        entry = {
            'class_id': yolo_ann['class_id'],
            'bbox': yolo_bbox,
            'yolo_ann': yolo_ann,
            'attributes': {},
            'label': '',
            'match_iou': best_iou
        }
        
        if best_json_idx >= 0 and best_iou > 0.3:
            json_obj = json_objects[best_json_idx]
            entry['attributes'] = json_obj['attributes']
            entry['label'] = json_obj['label']
            used_json.add(best_json_idx)
        
        merged.append(entry)
    
    return merged


def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def discover_attributes(
    json_dir: Optional[str] = None,
    master_json: Optional[str] = None
) -> Dict[str, Set[str]]:
    """
    Discover all unique attribute names and their possible values.
    
    Scans JSON files to find what attributes exist and what values they take.
    Returns: {attribute_name: {value1, value2, ...}}
    """
    attributes: Dict[str, Set[str]] = {}
    
    # Scan per-image JSON files
    if json_dir and os.path.isdir(json_dir):
        for fname in os.listdir(json_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(json_dir, fname)
            try:
                data = load_json_attributes(fpath)
                objects = parse_xanylabeling_json(data)
                for obj in objects:
                    for attr_name, attr_val in obj['attributes'].items():
                        if attr_name not in attributes:
                            attributes[attr_name] = set()
                        if attr_val:
                            attributes[attr_name].add(str(attr_val))
            except (json.JSONDecodeError, KeyError, Exception):
                continue
    
    # Scan master JSON
    if master_json and os.path.exists(master_json):
        try:
            with open(master_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different master JSON formats
            if isinstance(data, dict):
                for img_key, img_data in data.items():
                    if isinstance(img_data, dict):
                        # Format: {image_name: {objects: [{attributes: {...}}]}}
                        for obj in img_data.get('objects', []):
                            for attr_name, attr_val in obj.get('attributes', {}).items():
                                if attr_name not in attributes:
                                    attributes[attr_name] = set()
                                if attr_val:
                                    attributes[attr_name].add(str(attr_val))
                        
                        # Also check direct attributes on image level
                        for attr_name, attr_val in img_data.get('attributes', {}).items():
                            if attr_name not in attributes:
                                attributes[attr_name] = set()
                            if attr_val:
                                attributes[attr_name].add(str(attr_val))
            
            elif isinstance(data, list):
                # Format: [{image: "...", objects: [{attributes: {...}}]}]
                for item in data:
                    if isinstance(item, dict):
                        for obj in item.get('objects', []):
                            for attr_name, attr_val in obj.get('attributes', {}).items():
                                if attr_name not in attributes:
                                    attributes[attr_name] = set()
                                if attr_val:
                                    attributes[attr_name].add(str(attr_val))
        except (json.JSONDecodeError, Exception):
            pass
    
    # Convert sets to sorted lists for JSON serialization
    return {k: sorted(v) for k, v in attributes.items()}
