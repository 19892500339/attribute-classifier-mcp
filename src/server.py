"""Attribute Classifier MCP Server.

Provides tools for:
1. Cropping objects from images using YOLO bbox annotations
2. Organizing crops by JSON attributes for classification datasets
3. Training CNN models per attribute
4. Running inference pipeline: YOLO detect → crop → CNN classify → JSON
5. Managing and replacing attribute classification models
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.utils.config import get_config
from src.utils.device_manager import DeviceManager, get_device_manager
from src.utils.data_loader import (
    load_classes,
    load_yolo_annotations,
    load_json_attributes,
    discover_attributes,
    parse_xanylabeling_json
)
from src.utils.cropper import (
    crop_single_image,
    crop_dataset,
    organize_by_attribute
)
from src.models.cnn_trainer import (
    train_attribute_model,
    train_all_attributes
)
from src.models.inference import (
    AttributeClassifier,
    ModelRegistry,
    InferencePipeline
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('attribute-classifier-mcp')

# Global state
config = get_config()
registry: Optional[ModelRegistry] = None
pipeline: Optional[InferencePipeline] = None


def get_registry() -> ModelRegistry:
    global registry
    if registry is None:
        models_dir = config.get('paths.models_dir', './models')
        device = config.get('training.device', 'auto')
        registry = ModelRegistry(models_dir, device)
    return registry


def get_pipeline() -> InferencePipeline:
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline(
            yolo_model_path=config.get('yolo.model', 'yolov8n.pt'),
            models_dir=config.get('paths.models_dir', './models'),
            master_json=config.get('paths.master_json'),
            confidence=config.get('yolo.confidence', 0.5),
            iou=config.get('yolo.iou', 0.45),
            device=config.get('training.device', 'auto'),
            target_classes=config.get('yolo.target_classes') or None
        )
    return pipeline


# Create MCP server
app = Server("attribute-classifier-mcp")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="crop_objects",
            description="Crop all annotated objects from images using YOLO txt bbox annotations. "
                        "Reads images directory + labels directory, crops each object by its bbox, "
                        "and saves to output directory organized by class.",
            inputSchema={
                "type": "object",
                "properties": {
                    "images_dir": {"type": "string", "description": "Directory containing source images"},
                    "labels_dir": {"type": "string", "description": "Directory containing YOLO txt annotation files"},
                    "json_dir": {"type": "string", "description": "Optional: Directory with per-image JSON attribute files (X-AnyLabeling format)"},
                    "output_dir": {"type": "string", "description": "Output directory for cropped images (default: ./data/cropped)"},
                    "classes_file": {"type": "string", "description": "Optional: Path to classes.txt with class names"},
                    "padding": {"type": "integer", "description": "Padding around bbox in pixels (default: 5)"},
                    "min_size": {"type": "integer", "description": "Minimum crop dimension in pixels (default: 32)"}
                },
                "required": ["images_dir", "labels_dir"]
            }
        ),
        Tool(
            name="organize_dataset",
            description="Organize cropped images into classification dataset folders by a specific attribute value. "
                        "For example, organize by 'material' to create folders: leather/, wood/, fabric/. "
                        "Each attribute creates a separate dataset for training a dedicated CNN model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cropped_dir": {"type": "string", "description": "Directory with cropped images (output of crop_objects)"},
                    "attribute_name": {"type": "string", "description": "Attribute to organize by (e.g. 'material', 'has_armrest')"},
                    "output_dir": {"type": "string", "description": "Output dataset directory"},
                    "json_dir": {"type": "string", "description": "Optional: Directory with per-image JSON attribute files"},
                    "master_json": {"type": "string", "description": "Optional: Path to master JSON with all attributes"},
                    "target_size": {"type": "array", "items": {"type": "integer"}, "description": "Resize images to [width, height] (default: [224, 224])"}
                },
                "required": ["cropped_dir", "attribute_name"]
            }
        ),
        Tool(
            name="discover_attributes",
            description="Discover all unique attribute names and their possible values from JSON annotation files. "
                        "Scans X-AnyLabeling JSON files or a master JSON to find what attributes exist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "json_dir": {"type": "string", "description": "Directory with per-image JSON attribute files"},
                    "master_json": {"type": "string", "description": "Path to master JSON file"}
                }
            }
        ),
        Tool(
            name="train_attribute_model",
            description="Train a CNN classifier for a specific attribute. Uses a dataset organized by attribute values "
                        "(folders named by value, containing images). Supports multiple CNN backbones.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_dir": {"type": "string", "description": "Dataset directory with value subfolders (e.g. leather/, wood/)"},
                    "attribute_name": {"type": "string", "description": "Name of the attribute being trained"},
                    "backbone": {"type": "string", "enum": ["resnet18", "resnet34", "resnet50", "mobilenet_v2", "efficientnet_b0"], "description": "CNN backbone (default: resnet18)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default: 50)"},
                    "batch_size": {"type": "integer", "description": "Batch size (default: 32)"},
                    "learning_rate": {"type": "number", "description": "Learning rate (default: 0.001)"},
                    "val_split": {"type": "number", "description": "Validation split ratio (default: 0.2)"},
                    "pretrained": {"type": "boolean", "description": "Use pretrained weights (default: true)"},
                    "device": {"type": "string", "description": "Device: auto, cpu, cuda (default: auto)"},
                    "early_stopping": {"type": "integer", "description": "Stop after N epochs without improvement (default: 10)"},
                    "output_dir": {"type": "string", "description": "Where to save the model (default: ./models)"}
                },
                "required": ["dataset_dir", "attribute_name"]
            }
        ),
        Tool(
            name="train_all_attributes",
            description="Train CNN classifiers for ALL attribute datasets found under a base directory. "
                        "Each subdirectory is treated as a separate attribute dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "datasets_dir": {"type": "string", "description": "Base directory containing attribute dataset subdirectories"},
                    "attributes": {"type": "array", "items": {"type": "string"}, "description": "Optional: specific attributes to train (default: all found)"},
                    "backbone": {"type": "string", "description": "CNN backbone (default: resnet18)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default: 50)"},
                    "batch_size": {"type": "integer", "description": "Batch size (default: 32)"},
                    "output_dir": {"type": "string", "description": "Where to save models (default: ./models)"}
                },
                "required": ["datasets_dir"]
            }
        ),
        Tool(
            name="detect_and_classify",
            description="Full inference pipeline: YOLO detection → crop each detected object → classify through "
                        "attribute CNN models → return structured JSON with all attribute predictions per object.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to input image"},
                    "attributes": {"type": "array", "items": {"type": "string"}, "description": "Optional: specific attributes to classify (default: all registered)"},
                    "confidence": {"type": "number", "description": "YOLO detection confidence threshold (default: 0.5)"},
                    "yolo_model": {"type": "string", "description": "Optional: override YOLO model path"}
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="classify_crop",
            description="Classify a pre-cropped object image through attribute CNN models. "
                        "Skips the YOLO detection step; directly runs attribute classification.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to cropped object image"},
                    "attributes": {"type": "array", "items": {"type": "string"}, "description": "Optional: specific attributes to classify (default: all registered)"}
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="batch_detect_and_classify",
            description="Run the full detection+classification pipeline on multiple images. "
                        "Returns a list of results, one per image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_paths": {"type": "array", "items": {"type": "string"}, "description": "List of image file paths"},
                    "image_dir": {"type": "string", "description": "Alternative: process all images in a directory"},
                    "attributes": {"type": "array", "items": {"type": "string"}, "description": "Optional: specific attributes to classify"}
                }
            }
        ),
        Tool(
            name="register_model",
            description="Register or replace a CNN model for a specific attribute. "
                        "Allows hot-swapping models without restarting the server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "attribute_name": {"type": "string", "description": "Attribute name (e.g. 'material')"},
                    "model_path": {"type": "string", "description": "Path to .pth model file"}
                },
                "required": ["attribute_name", "model_path"]
            }
        ),
        Tool(
            name="unregister_model",
            description="Remove a model from the registry for a specific attribute.",
            inputSchema={
                "type": "object",
                "properties": {
                    "attribute_name": {"type": "string", "description": "Attribute name to remove"}
                },
                "required": ["attribute_name"]
            }
        ),
        Tool(
            name="list_models",
            description="List all registered and auto-discovered attribute classification models with metadata.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_model_info",
            description="Get detailed information about a specific attribute's classification model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "attribute_name": {"type": "string", "description": "Attribute name"}
                },
                "required": ["attribute_name"]
            }
        ),
        Tool(
            name="update_config",
            description="Update server configuration (paths, training params, YOLO settings, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Config key path (e.g. 'training.epochs', 'yolo.model')"},
                    "value": {"description": "New value"}
                },
                "required": ["key", "value"]
            }
        ),
        Tool(
            name="get_config",
            description="Get current server configuration or a specific config value.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Optional: specific config key path (omit for full config)"}
                }
            }
        ),
        Tool(
            name="set_class_attributes",
            description="Configure which attributes to classify for each detected class. "
                        "E.g. {'chair': ['material', 'has_armrest'], 'sofa': ['material', 'seat_count']}",
            inputSchema={
                "type": "object",
                "properties": {
                    "mapping": {
                        "type": "object",
                        "description": "Mapping of class names to attribute lists",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "required": ["mapping"]
            }
        ),
        Tool(
            name="full_pipeline",
            description="Run the complete pipeline from raw data to trained models: "
                        "1) Crop objects from images, 2) Organize by each attribute, "
                        "3) Train CNN models for all attributes. One-command setup.",
            inputSchema={
                "type": "object",
                "properties": {
                    "images_dir": {"type": "string", "description": "Directory with source images"},
                    "labels_dir": {"type": "string", "description": "Directory with YOLO txt annotations"},
                    "json_dir": {"type": "string", "description": "Directory with per-image JSON attribute files"},
                    "master_json": {"type": "string", "description": "Path to master JSON with attributes"},
                    "classes_file": {"type": "string", "description": "Path to classes.txt"},
                    "attributes": {"type": "array", "items": {"type": "string"}, "description": "Specific attributes to process (default: all discovered)"},
                    "output_base": {"type": "string", "description": "Base output directory (default: ./data)"},
                    "backbone": {"type": "string", "description": "CNN backbone (default: resnet18)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default: 50)"}
                },
                "required": ["images_dir", "labels_dir"]
            }
        ),
        Tool(
            name="get_system_info",
            description="Get comprehensive system hardware info: CPU cores, RAM, GPU VRAM, "
                        "CUDA availability, PyTorch version, active device, and memory usage.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="set_device",
            description="Switch compute device between CPU and GPU, with resource limit. "
                        "'cpu': force all compute to CPU, maximize RAM + threads. "
                        "'cuda': force all compute to GPU, maximize VRAM + CUDA. "
                        "'auto': auto-detect best. max_resource_percent caps usage (default 60%).",
            inputSchema={
                "type": "object",
                "properties": {
                    "device": {"type": "string", "enum": ["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                               "description": "Device: auto, cpu, cuda, cuda:0, cuda:1"},
                    "max_resource_percent": {"type": "number",
                                             "description": "Max percentage of resources to use (1-100, default: 60)"}
                },
                "required": ["device"]
            }
        ),
        Tool(
            name="estimate_batch_size",
            description="Estimate the maximum training batch size that fits in available CPU RAM or GPU VRAM "
                        "for a given backbone and image size, respecting the resource limit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backbone": {"type": "string", "description": "CNN backbone (default: resnet18)"},
                    "image_size": {"type": "integer", "description": "Input image size in pixels (default: 224)"},
                    "num_classes": {"type": "integer", "description": "Number of classes (default: 10)"}
                }
            }
        ),
        Tool(
            name="adaptive_training_config",
            description="Auto-tune ALL training parameters based on hardware profiling. "
                        "Checks CPU/RAM/GPU performance and returns optimal batch_size, image_size, "
                        "num_workers, learning_rate, epochs, accumulation_steps — all capped at "
                        "max_resource_percent (default 60%) of available resources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backbone": {"type": "string", "description": "CNN backbone (default: resnet18)"},
                    "num_classes": {"type": "integer", "description": "Number of output classes (default: 10)"},
                    "dataset_size": {"type": "integer", "description": "Total images in dataset (default: 1000)"},
                    "image_size": {"type": "integer", "description": "Base image size (default: 224)"},
                    "device": {"type": "string", "description": "Device override (default: current device)"},
                    "max_resource_percent": {"type": "number",
                                             "description": "Max resource usage 1-100% (default: 60)"}
                }
            }
        ),
        Tool(
            name="memory_summary",
            description="Get current memory usage summary: CPU RAM, GPU VRAM allocated/free, process memory.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global pipeline
    
    try:
        result = await _handle_tool(name, arguments)
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "tool": name}, ensure_ascii=False)
        )]


async def _handle_tool(name: str, args: Dict[str, Any]) -> Any:
    """Route and execute tool calls."""
    
    if name == "crop_objects":
        return await asyncio.to_thread(
            crop_dataset,
            images_dir=args['images_dir'],
            labels_dir=args['labels_dir'],
            json_dir=args.get('json_dir'),
            output_dir=args.get('output_dir', config.get('paths.cropped_dir', './data/cropped')),
            classes_file=args.get('classes_file', config.get('paths.classes_file')),
            padding=args.get('padding', config.get('crop.padding', 5)),
            min_size=args.get('min_size', config.get('crop.min_size', 32))
        )
    
    elif name == "organize_dataset":
        target_size = args.get('target_size', config.get('crop.target_size', [224, 224]))
        attr_name = args['attribute_name']
        default_output = os.path.join(
            config.get('paths.datasets_dir', './data/datasets'),
            attr_name
        )
        return await asyncio.to_thread(
            organize_by_attribute,
            cropped_dir=args['cropped_dir'],
            json_dir=args.get('json_dir', config.get('paths.json_dir')),
            master_json=args.get('master_json', config.get('paths.master_json')),
            attribute_name=attr_name,
            output_dir=args.get('output_dir', default_output),
            target_size=tuple(target_size)
        )
    
    elif name == "discover_attributes":
        return await asyncio.to_thread(
            discover_attributes,
            json_dir=args.get('json_dir', config.get('paths.json_dir')),
            master_json=args.get('master_json', config.get('paths.master_json'))
        )
    
    elif name == "train_attribute_model":
        train_config = config.get('training', {})
        return await asyncio.to_thread(
            train_attribute_model,
            dataset_dir=args['dataset_dir'],
            attribute_name=args['attribute_name'],
            output_dir=args.get('output_dir', config.get('paths.models_dir', './models')),
            backbone=args.get('backbone', train_config.get('backbone', 'resnet18')),
            epochs=args.get('epochs', train_config.get('epochs', 50)),
            batch_size=args.get('batch_size', train_config.get('batch_size', 32)),
            learning_rate=args.get('learning_rate', train_config.get('learning_rate', 0.001)),
            val_split=args.get('val_split', train_config.get('val_split', 0.2)),
            pretrained=args.get('pretrained', train_config.get('pretrained', True)),
            device=args.get('device', train_config.get('device', 'auto')),
            early_stopping=args.get('early_stopping', train_config.get('early_stopping', 10)),
            target_size=tuple(config.get('crop.target_size', [224, 224])),
            augmentation=train_config.get('augmentation', {})
        )
    
    elif name == "train_all_attributes":
        train_config = config.get('training', {})
        return await asyncio.to_thread(
            train_all_attributes,
            datasets_base_dir=args['datasets_dir'],
            output_dir=args.get('output_dir', config.get('paths.models_dir', './models')),
            attributes=args.get('attributes'),
            backbone=args.get('backbone', train_config.get('backbone', 'resnet18')),
            epochs=args.get('epochs', train_config.get('epochs', 50)),
            batch_size=args.get('batch_size', train_config.get('batch_size', 32)),
            learning_rate=train_config.get('learning_rate', 0.001),
            val_split=train_config.get('val_split', 0.2),
            pretrained=train_config.get('pretrained', True),
            device=train_config.get('device', 'auto'),
            early_stopping=train_config.get('early_stopping', 10),
            target_size=tuple(config.get('crop.target_size', [224, 224])),
            augmentation=train_config.get('augmentation', {})
        )
    
    elif name == "detect_and_classify":
        p = get_pipeline()
        if args.get('confidence'):
            p.confidence = args['confidence']
        if args.get('yolo_model'):
            from ultralytics import YOLO
            p.yolo = YOLO(args['yolo_model'])
        return await asyncio.to_thread(
            p.detect_and_classify,
            image_path=args['image_path'],
            attributes=args.get('attributes')
        )
    
    elif name == "classify_crop":
        p = get_pipeline()
        return await asyncio.to_thread(
            p.classify_crop,
            image_path=args['image_path'],
            attributes=args.get('attributes')
        )
    
    elif name == "batch_detect_and_classify":
        p = get_pipeline()
        image_paths = args.get('image_paths', [])
        if not image_paths and args.get('image_dir'):
            img_dir = args['image_dir']
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            image_paths = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if Path(f).suffix.lower() in exts
            ])
        return await asyncio.to_thread(
            p.batch_detect_and_classify,
            image_paths=image_paths,
            attributes=args.get('attributes')
        )
    
    elif name == "register_model":
        reg = get_registry()
        reg.register_model(args['attribute_name'], args['model_path'])
        # Reset pipeline to pick up new model
        pipeline = None
        return {
            'success': True,
            'attribute': args['attribute_name'],
            'model_path': args['model_path'],
            'message': f"Model registered for attribute '{args['attribute_name']}'"
        }
    
    elif name == "unregister_model":
        reg = get_registry()
        reg.unregister_model(args['attribute_name'])
        pipeline = None
        return {
            'success': True,
            'attribute': args['attribute_name'],
            'message': f"Model unregistered for attribute '{args['attribute_name']}'"
        }
    
    elif name == "list_models":
        reg = get_registry()
        return reg.list_models()
    
    elif name == "get_model_info":
        reg = get_registry()
        classifier = reg.get_classifier(args['attribute_name'])
        if classifier:
            return classifier.info()
        return {'error': f"No model found for attribute: {args['attribute_name']}"}
    
    elif name == "update_config":
        config.set(args['key'], args['value'])
        # Reset pipeline/registry to pick up changes
        global registry
        pipeline = None
        registry = None
        return {
            'success': True,
            'key': args['key'],
            'value': args['value']
        }
    
    elif name == "get_config":
        key = args.get('key')
        if key:
            return {key: config.get(key)}
        return config.to_dict()
    
    elif name == "set_class_attributes":
        p = get_pipeline()
        p.set_class_attributes(args['mapping'])
        return {
            'success': True,
            'mapping': args['mapping']
        }
    
    elif name == "full_pipeline":
        return await _run_full_pipeline(args)
    
    elif name == "get_system_info":
        dm = get_device_manager(config.get('training.device', 'auto'))
        return dm.get_system_info()
    
    elif name == "set_device":
        new_device = args['device']
        max_pct = args.get('max_resource_percent', 60.0)
        dm = get_device_manager(new_device, max_pct)
        config.set('training.device', new_device)
        # Reset pipeline and registry to use new device
        pipeline = None
        registry = None
        # Apply dedicated mode
        if dm.is_gpu:
            mode_info = dm.apply_gpu_dedicated_mode()
        else:
            mode_info = dm.apply_cpu_dedicated_mode()
        info = dm.get_system_info()
        return {
            'success': True,
            'device': str(dm.device),
            'is_gpu': dm.is_gpu,
            'is_cpu': dm.is_cpu,
            'mode': mode_info,
            'system_info': info
        }
    
    elif name == "estimate_batch_size":
        from src.models.cnn_trainer import get_backbone
        dm = get_device_manager(config.get('training.device', 'auto'))
        backbone_name = args.get('backbone', config.get('training.backbone', 'resnet18'))
        img_size = args.get('image_size', 224)
        num_classes = args.get('num_classes', 10)
        model = get_backbone(backbone_name, num_classes, pretrained=False)
        model = model.to(dm.device)
        max_batch = dm.estimate_max_batch_size(model, input_size=(3, img_size, img_size))
        del model
        dm.clear_memory()
        return {
            'device': str(dm.device),
            'backbone': backbone_name,
            'image_size': img_size,
            'estimated_max_batch_size': max_batch,
            'recommended_batch_size': min(max_batch, 64)
        }
    
    elif name == "adaptive_training_config":
        dm = get_device_manager(
            args.get('device', config.get('training.device', 'auto')),
            args.get('max_resource_percent', 60.0)
        )
        adaptive = dm.adaptive_training_config(
            backbone=args.get('backbone', config.get('training.backbone', 'resnet18')),
            num_classes=args.get('num_classes', 10),
            dataset_size=args.get('dataset_size', 1000),
            base_image_size=args.get('image_size', 224),
            max_resource_percent=args.get('max_resource_percent')
        )
        return adaptive
    
    elif name == "memory_summary":
        dm = get_device_manager(config.get('training.device', 'auto'))
        return {
            'summary': dm.memory_summary(),
            'device': str(dm.device)
        }
    
    else:
        return {'error': f'Unknown tool: {name}'}


async def _run_full_pipeline(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete pipeline from raw data to trained models."""
    results = {'steps': {}}
    output_base = args.get('output_base', './data')
    
    # Step 1: Crop objects
    logger.info("Step 1: Cropping objects from images...")
    cropped_dir = os.path.join(output_base, 'cropped')
    crop_result = await asyncio.to_thread(
        crop_dataset,
        images_dir=args['images_dir'],
        labels_dir=args['labels_dir'],
        json_dir=args.get('json_dir'),
        output_dir=cropped_dir,
        classes_file=args.get('classes_file'),
        padding=config.get('crop.padding', 5),
        min_size=config.get('crop.min_size', 32)
    )
    results['steps']['crop'] = crop_result
    logger.info(f"Cropped {crop_result['total_crops']} objects from {crop_result['images_processed']} images")
    
    # Step 2: Discover attributes
    logger.info("Step 2: Discovering attributes...")
    attrs = await asyncio.to_thread(
        discover_attributes,
        json_dir=args.get('json_dir'),
        master_json=args.get('master_json')
    )
    
    target_attrs = args.get('attributes') or list(attrs.keys())
    results['steps']['discovered_attributes'] = {
        k: list(v) if isinstance(v, (set, list)) else v
        for k, v in attrs.items()
    }
    logger.info(f"Found attributes: {target_attrs}")
    
    # Step 3: Organize datasets per attribute
    logger.info("Step 3: Organizing datasets by attribute...")
    datasets_dir = os.path.join(output_base, 'datasets')
    organize_results = {}
    
    for attr_name in target_attrs:
        attr_output = os.path.join(datasets_dir, attr_name)
        org_result = await asyncio.to_thread(
            organize_by_attribute,
            cropped_dir=cropped_dir,
            json_dir=args.get('json_dir'),
            master_json=args.get('master_json'),
            attribute_name=attr_name,
            output_dir=attr_output,
            target_size=tuple(config.get('crop.target_size', [224, 224]))
        )
        organize_results[attr_name] = org_result
        logger.info(f"Organized {org_result['total_images']} images for attribute '{attr_name}'")
    
    results['steps']['organize'] = organize_results
    
    # Step 4: Train models
    logger.info("Step 4: Training CNN models...")
    train_config = config.get('training', {})
    train_results = await asyncio.to_thread(
        train_all_attributes,
        datasets_base_dir=datasets_dir,
        output_dir=config.get('paths.models_dir', './models'),
        attributes=target_attrs,
        backbone=args.get('backbone', train_config.get('backbone', 'resnet18')),
        epochs=args.get('epochs', train_config.get('epochs', 50)),
        batch_size=train_config.get('batch_size', 32),
        learning_rate=train_config.get('learning_rate', 0.001),
        val_split=train_config.get('val_split', 0.2),
        pretrained=train_config.get('pretrained', True),
        device=train_config.get('device', 'auto'),
        early_stopping=train_config.get('early_stopping', 10),
        target_size=tuple(config.get('crop.target_size', [224, 224])),
        augmentation=train_config.get('augmentation', {})
    )
    
    # Sanitize history for JSON serialization
    for attr, res in train_results.items():
        if 'history' in res:
            del res['history']  # Remove large history for output
    
    results['steps']['training'] = train_results
    
    # Summary
    results['summary'] = {
        'total_crops': crop_result['total_crops'],
        'attributes_trained': len([r for r in train_results.values() if r.get('success')]),
        'attributes_failed': len([r for r in train_results.values() if not r.get('success')]),
        'models': {
            attr: {
                'accuracy': res.get('best_val_acc', 0),
                'classes': res.get('class_names', []),
                'model_path': res.get('model_path', '')
            }
            for attr, res in train_results.items()
            if res.get('success')
        }
    }
    
    return results


async def main():
    """Start the MCP server."""
    # Check for config file argument
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if os.path.exists(config_path):
            get_config(config_path)
            logger.info(f"Loaded config from: {config_path}")
    
    logger.info("Starting Attribute Classifier MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
