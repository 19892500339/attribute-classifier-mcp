# Attribute Classifier MCP Server

> 🎯 **YOLO Detection → Object Cropping → CNN Attribute Classification → JSON Results**

A Model Context Protocol (MCP) server that combines YOLO object detection with per-attribute CNN classifiers. Detects objects in images, crops them, and classifies each object's visual attributes (material, color, has_armrest, etc.) using dedicated CNN models.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📦 **Object Cropping** | Crop objects from images using YOLO txt bbox annotations |
| 🗂️ **Dataset Organization** | Organize cropped images by attribute value for CNN training |
| 🧠 **CNN Training** | Train dedicated CNN classifiers per attribute (ResNet, MobileNet, EfficientNet) |
| 🔍 **Full Pipeline Inference** | YOLO detect → crop → multi-attribute CNN classify → JSON output |
| 🔄 **Hot-swap Models** | Register/replace attribute models without restarting |
| 📋 **Attribute Discovery** | Auto-discover attributes from X-AnyLabeling JSON annotations |
| ⚡ **Batch Processing** | Process multiple images in one call |
| ⚙️ **Configurable** | YAML config for paths, training params, YOLO settings |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/attribute-classifier-mcp.git
cd attribute-classifier-mcp

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Configure MCP Client

Add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "attribute-classifier": {
      "command": "python",
      "args": ["src/server.py"],
      "cwd": "/path/to/attribute-classifier-mcp"
    }
  }
}
```

### With Docker

```bash
docker build -t attribute-classifier-mcp .
docker run -v /your/data:/app/data -v /your/models:/app/models attribute-classifier-mcp
```

## 📊 Complete Workflow

### Step 1: Prepare Your Data

```
data/
├── images/          # Original images (.jpg, .png)
├── labels/          # YOLO txt annotations (one per image)
│   └── img001.txt   # "0 0.5 0.5 0.3 0.4" (class cx cy w h)
├── jsons/           # X-AnyLabeling JSON attributes (one per image)
│   └── img001.json  # {shapes: [{label, attributes: {material: "leather"}}]}
├── classes.txt      # Class names (one per line)
└── attributes.json  # Optional: master JSON with all attributes
```

### Step 2: Crop Objects

Use the `crop_objects` tool to crop all annotated objects from images:

```json
{
  "tool": "crop_objects",
  "arguments": {
    "images_dir": "./data/images",
    "labels_dir": "./data/labels",
    "json_dir": "./data/jsons",
    "classes_file": "./data/classes.txt"
  }
}
```

### Step 3: Organize by Attribute

Organize crops into classification datasets:

```json
{
  "tool": "organize_dataset",
  "arguments": {
    "cropped_dir": "./data/cropped",
    "attribute_name": "material",
    "master_json": "./data/attributes.json"
  }
}
```

This creates:
```
data/datasets/material/
├── leather/     # Images of leather objects
├── wood/        # Images of wooden objects
├── fabric/      # Images of fabric objects
└── metal/       # Images of metal objects
```

### Step 4: Train Models

Train a CNN for each attribute:

```json
{
  "tool": "train_attribute_model",
  "arguments": {
    "dataset_dir": "./data/datasets/material",
    "attribute_name": "material",
    "backbone": "resnet18",
    "epochs": 50
  }
}
```

Or train all attributes at once:

```json
{
  "tool": "train_all_attributes",
  "arguments": {
    "datasets_dir": "./data/datasets"
  }
}
```

### Step 5: Inference

Run the full pipeline on new images:

```json
{
  "tool": "detect_and_classify",
  "arguments": {
    "image_path": "./test.jpg"
  }
}
```

**Output:**
```json
{
  "image": "test.jpg",
  "image_size": {"width": 1920, "height": 1080},
  "total_objects": 2,
  "detections": [
    {
      "index": 0,
      "class_name": "chair",
      "confidence": 0.92,
      "bbox": {"x1": 100, "y1": 200, "x2": 400, "y2": 600},
      "attributes": {
        "material": {
          "attribute": "material",
          "predicted_value": "leather",
          "confidence": 0.95,
          "all_probabilities": {
            "leather": 0.95,
            "wood": 0.03,
            "fabric": 0.02
          }
        },
        "has_armrest": {
          "attribute": "has_armrest",
          "predicted_value": "yes",
          "confidence": 0.88,
          "all_probabilities": {"yes": 0.88, "no": 0.12}
        }
      }
    }
  ]
}
```

## 🛠️ All MCP Tools (16 total)

### Data Preparation
| Tool | Description |
|------|-------------|
| `crop_objects` | Crop objects from images using YOLO txt bbox annotations |
| `organize_dataset` | Organize crops by attribute value into classification folders |
| `discover_attributes` | Discover all attribute names and values from JSON annotations |

### Training
| Tool | Description |
|------|-------------|
| `train_attribute_model` | Train a CNN classifier for one attribute |
| `train_all_attributes` | Train CNN classifiers for all discovered attributes |

### Inference
| Tool | Description |
|------|-------------|
| `detect_and_classify` | Full pipeline: YOLO → crop → CNN classify → JSON |
| `classify_crop` | Classify a pre-cropped image through attribute models |
| `batch_detect_and_classify` | Process multiple images through the full pipeline |

### Model Management
| Tool | Description |
|------|-------------|
| `register_model` | Register/replace a CNN model for an attribute |
| `unregister_model` | Remove a model from the registry |
| `list_models` | List all registered attribute models |
| `get_model_info` | Get detailed info about a specific model |

### Configuration
| Tool | Description |
|------|-------------|
| `update_config` | Update server configuration |
| `get_config` | Get current configuration |
| `set_class_attributes` | Map detected classes to their attributes |
| `full_pipeline` | Run complete pipeline: crop → organize → train |

## 🔧 Configuration

Edit `config/default_config.yaml` or pass a custom config:

```bash
python src/server.py /path/to/my_config.yaml
```

### Key Config Options

```yaml
# Paths
paths:
  images_dir: "./data/images"
  labels_dir: "./data/labels"
  json_dir: "./data/jsons"
  master_json: "./data/attributes.json"

# Training
training:
  backbone: "resnet18"    # resnet18/34/50, mobilenet_v2, efficientnet_b0
  epochs: 50
  batch_size: 32
  device: "auto"          # auto, cpu, cuda

# YOLO
yolo:
  model: "yolov8n.pt"
  confidence: 0.5
```

## 🔄 Hot-swap Models

Replace any attribute's model at runtime:

```json
{
  "tool": "register_model",
  "arguments": {
    "attribute_name": "material",
    "model_path": "./models/material_efficientnet_b0_best.pth"
  }
}
```

## 📁 Supported Annotation Formats

### YOLO TXT (for bounding boxes)
```
0 0.5125 0.4833 0.3250 0.4167
1 0.2500 0.7500 0.2000 0.3000
```

### X-AnyLabeling JSON (for attributes)
```json
{
  "shapes": [
    {
      "label": "chair",
      "points": [[100, 200], [400, 600]],
      "shape_type": "rectangle",
      "flags": {"has_armrest": true},
      "attributes": {"material": "leather", "seat_count": "1"}
    }
  ]
}
```

## 📜 License

MIT License
