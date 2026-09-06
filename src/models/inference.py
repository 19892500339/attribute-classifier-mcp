"""Inference pipeline: YOLO detection → Crop → CNN attribute classification → JSON results."""
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class AttributeClassifier:
    """
    Loads and runs a single attribute CNN classifier model.
    Models are .pth files saved by cnn_trainer.train_attribute_model.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = model_path
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        self.backbone_name = checkpoint['backbone']
        self.attribute_name = checkpoint['attribute_name']
        self.target_size = tuple(checkpoint.get('target_size', (224, 224)))
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        
        # Rebuild model architecture
        from .cnn_trainer import get_backbone
        self.model = get_backbone(self.backbone_name, self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Inference transform
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify a single image.
        
        Args:
            image: PIL Image (cropped object)
        
        Returns:
            Dict with predicted class, confidence, and all probabilities
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        pred_idx = probs.argmax().item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()
        
        all_probs = {
            self.class_names[i]: round(probs[i].item(), 4)
            for i in range(self.num_classes)
        }
        
        return {
            'attribute': self.attribute_name,
            'predicted_value': pred_class,
            'confidence': round(confidence, 4),
            'all_probabilities': all_probs
        }
    
    def info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            'attribute_name': self.attribute_name,
            'backbone': self.backbone_name,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'target_size': list(self.target_size),
            'best_val_acc': self.best_val_acc,
            'model_path': self.model_path,
            'device': str(self.device)
        }


class ModelRegistry:
    """
    Registry for managing multiple attribute classification models.
    Supports loading, replacing, and querying models.
    """
    
    def __init__(self, models_dir: str = './models', device: str = 'auto'):
        self.models_dir = models_dir
        self.device = device
        self.classifiers: Dict[str, AttributeClassifier] = {}
        self._config_path = os.path.join(models_dir, 'registry.json')
        self._model_config: Dict[str, str] = {}  # attribute -> model_path
        self._load_config()
    
    def _load_config(self):
        """Load model registry config."""
        if os.path.exists(self._config_path):
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._model_config = json.load(f)
    
    def _save_config(self):
        """Save model registry config."""
        os.makedirs(self.models_dir, exist_ok=True)
        with open(self._config_path, 'w', encoding='utf-8') as f:
            json.dump(self._model_config, f, ensure_ascii=False, indent=2)
    
    def register_model(self, attribute_name: str, model_path: str):
        """Register or replace a model for an attribute."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Unload existing model if loaded
        if attribute_name in self.classifiers:
            del self.classifiers[attribute_name]
        
        self._model_config[attribute_name] = model_path
        self._save_config()
    
    def unregister_model(self, attribute_name: str):
        """Remove a model from the registry."""
        if attribute_name in self.classifiers:
            del self.classifiers[attribute_name]
        if attribute_name in self._model_config:
            del self._model_config[attribute_name]
            self._save_config()
    
    def get_classifier(self, attribute_name: str) -> Optional[AttributeClassifier]:
        """Get a loaded classifier for an attribute (lazy loading)."""
        if attribute_name not in self.classifiers:
            model_path = self._model_config.get(attribute_name)
            if model_path is None:
                # Try auto-discovery
                model_path = self._auto_discover(attribute_name)
            if model_path and os.path.exists(model_path):
                self.classifiers[attribute_name] = AttributeClassifier(
                    model_path, device=self.device
                )
        return self.classifiers.get(attribute_name)
    
    def _auto_discover(self, attribute_name: str) -> Optional[str]:
        """Try to find a model file for the attribute in models_dir."""
        if not os.path.isdir(self.models_dir):
            return None
        for fname in os.listdir(self.models_dir):
            if fname.startswith(f"{attribute_name}_") and fname.endswith('.pth'):
                path = os.path.join(self.models_dir, fname)
                self._model_config[attribute_name] = path
                self._save_config()
                return path
        return None
    
    def list_models(self) -> Dict[str, Dict]:
        """List all registered models with their info."""
        result = {}
        
        # Include registered models
        for attr, path in self._model_config.items():
            info = {'model_path': path, 'loaded': attr in self.classifiers, 'exists': os.path.exists(path)}
            if attr in self.classifiers:
                info.update(self.classifiers[attr].info())
            elif os.path.exists(path):
                # Read metadata without loading the full model
                meta_path = path.replace('.pth', '').rsplit('_best', 1)[0]
                meta_json = f"{meta_path}_meta.json" if not meta_path.endswith('_meta') else f"{meta_path}.json"
                # Try alternative meta path patterns
                for mp in [path.replace('.pth', '_meta.json'), 
                           path.replace('_best.pth', '_meta.json')]:
                    if os.path.exists(mp):
                        with open(mp, 'r', encoding='utf-8') as f:
                            info['metadata'] = json.load(f)
                        break
            result[attr] = info
        
        # Also discover unregistered models
        if os.path.isdir(self.models_dir):
            for fname in os.listdir(self.models_dir):
                if fname.endswith('.pth') and '_best.pth' in fname:
                    parts = fname.replace('_best.pth', '').rsplit('_', 1)
                    if len(parts) == 2:
                        attr = parts[0]
                        if attr not in result:
                            path = os.path.join(self.models_dir, fname)
                            result[attr] = {
                                'model_path': path,
                                'loaded': False,
                                'exists': True,
                                'auto_discovered': True
                            }
        
        return result
    
    def get_registered_attributes(self) -> List[str]:
        """Get list of attributes that have registered models."""
        return list(self._model_config.keys())


class InferencePipeline:
    """
    Full inference pipeline:
    1. YOLO detection → find objects in image
    2. Crop each detected object
    3. Look up attribute models for each detected class
    4. Run each crop through attribute classifiers
    5. Return structured JSON results
    """
    
    def __init__(
        self,
        yolo_model_path: str = 'yolov8n.pt',
        models_dir: str = './models',
        master_json: Optional[str] = None,
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str = 'auto',
        target_classes: Optional[List[int]] = None
    ):
        self.confidence = confidence
        self.iou = iou
        self.target_classes = target_classes
        
        # Load YOLO model
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize model registry
        self.registry = ModelRegistry(models_dir, device)
        
        # Load master JSON for attribute-class mapping
        self.class_attributes: Dict[str, List[str]] = {}  # class_name -> [attribute_names]
        if master_json and os.path.exists(master_json):
            self._load_class_attributes(master_json)
    
    def _load_class_attributes(self, master_json: str):
        """Load which attributes apply to which classes from master JSON."""
        with open(master_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Collect all attributes per class label
        for img_key, img_data in data.items():
            if isinstance(img_data, dict):
                for obj in img_data.get('objects', []):
                    label = obj.get('label', '')
                    attrs = list(obj.get('attributes', {}).keys())
                    if label and attrs:
                        if label not in self.class_attributes:
                            self.class_attributes[label] = set()
                        self.class_attributes[label].update(attrs)
        
        # Convert sets to sorted lists
        self.class_attributes = {
            k: sorted(v) for k, v in self.class_attributes.items()
        }
    
    def set_class_attributes(self, mapping: Dict[str, List[str]]):
        """Manually set which attributes to classify for each detected class."""
        self.class_attributes = mapping
    
    def detect_and_classify(
        self,
        image_path: str,
        attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on an image.
        
        Args:
            image_path: Path to input image
            attributes: Override attribute list (None = use class_attributes mapping)
        
        Returns:
            JSON-serializable results dict
        """
        # Step 1: YOLO Detection
        results = self.yolo.predict(
            source=image_path,
            conf=self.confidence,
            iou=self.iou,
            classes=self.target_classes,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return {
                'image': image_path,
                'detections': [],
                'total_objects': 0,
                'message': 'No objects detected'
            }
        
        result = results[0]
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_w, img_h = img.size
        
        detections = []
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            
            # Step 2: Crop the detected object
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            
            detection = {
                'index': i,
                'class_name': cls_name,
                'class_id': cls_id,
                'confidence': round(conf, 4),
                'bbox': {
                    'x1': round(x1, 1),
                    'y1': round(y1, 1),
                    'x2': round(x2, 1),
                    'y2': round(y2, 1)
                },
                'attributes': {}
            }
            
            # Step 3: Determine which attributes to classify
            if attributes:
                attr_list = attributes
            elif cls_name in self.class_attributes:
                attr_list = self.class_attributes[cls_name]
            else:
                # Try all registered models
                attr_list = self.registry.get_registered_attributes()
            
            # Step 4: Run through each attribute classifier
            for attr_name in attr_list:
                classifier = self.registry.get_classifier(attr_name)
                if classifier is None:
                    detection['attributes'][attr_name] = {
                        'error': f'No model found for attribute: {attr_name}'
                    }
                    continue
                
                try:
                    pred = classifier.predict(crop)
                    detection['attributes'][attr_name] = pred
                except Exception as e:
                    detection['attributes'][attr_name] = {
                        'error': str(e)
                    }
            
            detections.append(detection)
        
        return {
            'image': os.path.basename(image_path),
            'image_size': {'width': img_w, 'height': img_h},
            'total_objects': len(detections),
            'detections': detections
        }
    
    def classify_crop(
        self,
        image_path: str,
        attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify a pre-cropped image through attribute models.
        Skip YOLO detection step.
        
        Args:
            image_path: Path to cropped object image
            attributes: Which attributes to classify (None = all registered)
        
        Returns:
            Classification results
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        attr_list = attributes or self.registry.get_registered_attributes()
        results = {}
        
        for attr_name in attr_list:
            classifier = self.registry.get_classifier(attr_name)
            if classifier is None:
                results[attr_name] = {'error': f'No model found for attribute: {attr_name}'}
                continue
            try:
                results[attr_name] = classifier.predict(img)
            except Exception as e:
                results[attr_name] = {'error': str(e)}
        
        return {
            'image': os.path.basename(image_path),
            'attributes': results
        }
    
    def batch_detect_and_classify(
        self,
        image_paths: List[str],
        attributes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run pipeline on multiple images."""
        return [
            self.detect_and_classify(p, attributes)
            for p in image_paths
        ]
