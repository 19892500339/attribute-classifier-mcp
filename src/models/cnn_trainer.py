"""CNN model training for attribute classification."""
import os
import json
import time
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RobustImageFolder(datasets.ImageFolder):
    """
    ImageFolder that skips corrupt/unreadable/wrong-format files instead of crashing.
    
    - Skips files that PIL cannot open
    - Skips files that fail during transform (e.g. truncated images)
    - Skips non-image files silently  
    - Logs skipped files for debugging
    - Reports total skipped count
    """
    
    SUPPORTED_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'
    }
    
    def __init__(self, root, transform=None, **kwargs):
        self.skipped_files = []
        self._pre_filter_root = root
        super().__init__(root, transform=transform, **kwargs)
        # Post-filter: remove entries that cannot be opened
        self._filter_bad_samples()
    
    def _filter_bad_samples(self):
        """Remove samples with unsupported extensions or that cannot be opened."""
        valid_samples = []
        for path, class_idx in self.samples:
            ext = os.path.splitext(path)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                self.skipped_files.append((path, f'unsupported extension: {ext}'))
                continue
            try:
                with Image.open(path) as img:
                    img.verify()  # quick integrity check
                valid_samples.append((path, class_idx))
            except Exception as e:
                self.skipped_files.append((path, str(e)))
        
        if self.skipped_files:
            logger.warning(
                f"RobustImageFolder: skipped {len(self.skipped_files)} bad files "
                f"out of {len(self.samples)} total in {self._pre_filter_root}"
            )
            for path, reason in self.skipped_files[:10]:
                logger.warning(f"  Skipped: {os.path.basename(path)} - {reason}")
            if len(self.skipped_files) > 10:
                logger.warning(f"  ... and {len(self.skipped_files) - 10} more")
        
        self.samples = valid_samples
        self.imgs = valid_samples  # ImageFolder alias
        self.targets = [s[1] for s in valid_samples]
    
    def __getitem__(self, index):
        """Load item with fallback: if transform fails, try next valid sample."""
        max_retries = min(5, len(self.samples))
        for attempt in range(max_retries):
            try:
                path, target = self.samples[(index + attempt) % len(self.samples)]
                sample = self.loader(path)
                if sample.mode != 'RGB':
                    sample = sample.convert('RGB')
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Error loading {path}: {e}, trying next")
                continue
        # Should not reach here, but fallback to parent
        return super().__getitem__(index)


def get_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Get a CNN backbone model.
    
    Supported: resnet18, resnet34, resnet50, mobilenet_v2, efficientnet_b0
    """
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    if name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet34':
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {name}. Choose from: resnet18, resnet34, resnet50, mobilenet_v2, efficientnet_b0")
    
    return model


def get_transforms(target_size: Tuple[int, int] = (224, 224), augment: bool = True, aug_config: Dict = None) -> Dict:
    """Get train and val transforms."""
    aug_config = aug_config or {}
    
    train_transforms_list = []
    if augment:
        if aug_config.get('horizontal_flip', True):
            train_transforms_list.append(transforms.RandomHorizontalFlip())
        if aug_config.get('vertical_flip', False):
            train_transforms_list.append(transforms.RandomVerticalFlip())
        rotation = aug_config.get('rotation', 15)
        if rotation > 0:
            train_transforms_list.append(transforms.RandomRotation(rotation))
        if aug_config.get('color_jitter', True):
            train_transforms_list.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
    
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        *train_transforms_list,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}


def train_attribute_model(
    dataset_dir: str,
    attribute_name: str,
    output_dir: str = "./models",
    backbone: str = "resnet18",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    pretrained: bool = True,
    device: str = "auto",
    num_workers: int = 4,
    early_stopping: int = 10,
    target_size: Tuple[int, int] = (224, 224),
    augmentation: Dict = None
) -> Dict[str, Any]:
    """
    Train a CNN classifier for a specific attribute.
    
    Args:
        dataset_dir: Directory organized as dataset_dir/{attribute_value}/images...
        attribute_name: Name of the attribute being classified
        output_dir: Where to save the trained model
        backbone: CNN backbone name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        val_split: Fraction of data for validation
        pretrained: Use ImageNet pretrained weights
        device: Compute device
        num_workers: DataLoader workers
        early_stopping: Stop after N epochs without improvement (0=disable)
        target_size: Input image size
        augmentation: Augmentation config dict
    
    Returns:
        Training results dict with metrics and model path
    """
    # Determine device with DeviceManager
    from src.utils.device_manager import DeviceManager
    dm = DeviceManager(device)
    device = dm.device
    
    print(f"Device: {device} | GPU: {dm.is_gpu}")
    if dm.is_gpu:
        print(dm.memory_summary())
    
    # Get transforms
    tx = get_transforms(target_size, augment=True, aug_config=augmentation or {})
    
    # Load dataset with RobustImageFolder (skips corrupt/bad files)
    full_dataset = RobustImageFolder(dataset_dir, transform=tx['train'])
    skipped_count = len(full_dataset.skipped_files)
    if skipped_count > 0:
        print(f"Skipped {skipped_count} bad/unreadable files during loading")
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    if num_classes < 2:
        return {
            'success': False,
            'error': f"Need at least 2 classes, found {num_classes}: {class_names}",
            'attribute': attribute_name
        }
    
    if len(full_dataset) < 2:
        return {
            'success': False,
            'error': f"Not enough valid images: {len(full_dataset)} (skipped {skipped_count})",
            'attribute': attribute_name
        }
    
    # --- Adaptive mode: auto-tune params if batch_size/epochs not explicitly set ---
    adaptive_config = dm.adaptive_training_config(
        backbone=backbone,
        num_classes=num_classes,
        dataset_size=len(full_dataset),
        base_image_size=target_size[0]
    )
    # Use adaptive values as smart defaults (explicit params override)
    if batch_size == 32:  # default => use adaptive
        batch_size = adaptive_config.get('batch_size', batch_size)
    if learning_rate == 0.001:  # default => use adaptive
        learning_rate = adaptive_config.get('learning_rate', learning_rate)
    
    print(f"Training config: batch_size={batch_size}, lr={learning_rate}, "
          f"img={target_size}, workers={num_workers}")
    print(f"Resource limit: {dm.max_resource_percent:.0f}% | "
          f"Estimated usage: {adaptive_config.get('resource_report', {}).get('estimated_usage_percent', '?')}%")
    
    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override val transform with RobustImageFolder
    val_dataset.dataset = RobustImageFolder(dataset_dir, transform=tx['val'])
    
    # Use DeviceManager for optimal DataLoader settings
    dl_kwargs = dm.get_dataloader_kwargs()
    if num_workers is not None and num_workers >= 0:
        dl_kwargs['num_workers'] = num_workers
        dl_kwargs['persistent_workers'] = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs
    )
    
    # Build model with device optimization
    model = get_backbone(backbone, num_classes, pretrained)
    model = dm.optimize_for_device(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
        
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if early_stopping > 0 and epochs_no_improve >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Save best model
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f"{attribute_name}_{backbone}_best.pth"
    model_path = os.path.join(output_dir, model_filename)
    
    save_data = {
        'model_state_dict': best_model_state or model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'backbone': backbone,
        'attribute_name': attribute_name,
        'target_size': target_size,
        'best_val_acc': best_val_acc,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'pretrained': pretrained
        }
    }
    torch.save(save_data, model_path)
    
    # Also save a metadata JSON
    meta_path = os.path.join(output_dir, f"{attribute_name}_{backbone}_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'attribute_name': attribute_name,
            'backbone': backbone,
            'class_names': class_names,
            'num_classes': num_classes,
            'target_size': list(target_size),
            'best_val_acc': best_val_acc,
            'total_epochs': len(history['train_loss']),
            'training_time_seconds': training_time,
            'train_samples': train_size,
            'val_samples': val_size,
            'model_file': model_filename
        }, f, ensure_ascii=False, indent=2)
    
    return {
        'success': True,
        'attribute': attribute_name,
        'model_path': model_path,
        'meta_path': meta_path,
        'class_names': class_names,
        'num_classes': num_classes,
        'best_val_acc': best_val_acc,
        'total_epochs': len(history['train_loss']),
        'training_time': training_time,
        'train_samples': train_size,
        'val_samples': val_size,
        'history': history,
        'device': str(device)
    }


def train_all_attributes(
    datasets_base_dir: str,
    output_dir: str = "./models",
    attributes: Optional[List[str]] = None,
    **train_kwargs
) -> Dict[str, Any]:
    """
    Train CNN classifiers for all attribute datasets found in the base directory.
    
    Each subdirectory under datasets_base_dir is treated as an attribute dataset:
    datasets_base_dir/
        material/
            leather/
            wood/
        has_armrest/
            yes/
            no/
    
    Args:
        datasets_base_dir: Base directory containing attribute subdirectories
        output_dir: Where to save trained models
        attributes: Specific attributes to train (None = all found)
        **train_kwargs: Passed to train_attribute_model
    
    Returns:
        Dict with results per attribute
    """
    results = {}
    
    # Find attribute datasets
    if attributes:
        attr_dirs = attributes
    else:
        attr_dirs = [
            d for d in os.listdir(datasets_base_dir)
            if os.path.isdir(os.path.join(datasets_base_dir, d))
        ]
    
    for attr_name in attr_dirs:
        dataset_dir = os.path.join(datasets_base_dir, attr_name)
        if not os.path.isdir(dataset_dir):
            results[attr_name] = {'success': False, 'error': f'Directory not found: {dataset_dir}'}
            continue
        
        # Check if it has subdirectories (class folders)
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if len(subdirs) < 2:
            results[attr_name] = {
                'success': False,
                'error': f'Need at least 2 value folders, found {len(subdirs)}: {subdirs}'
            }
            continue
        
        print(chr(10) + "=" * 60)
        print("Training model for attribute: " + attr_name)
        print("Classes: " + str(subdirs))
        print("=" * 60)
        
        result = train_attribute_model(
            dataset_dir=dataset_dir,
            attribute_name=attr_name,
            output_dir=output_dir,
            **train_kwargs
        )
        results[attr_name] = result
    
    return results
