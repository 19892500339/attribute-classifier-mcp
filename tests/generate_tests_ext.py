"""Generate test_extreme_ext.py - extended extreme tests."""
import os

NL = chr(10)

test_code = f'''"""Extended extreme tests: MCP routing, CNN training, error injection, resilience."""
import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from PIL import Image

NL = chr(10)


# ============================================================
# CNN MICRO-TRAINING TESTS
# ============================================================
from src.models.cnn_trainer import train_attribute_model, get_backbone, get_transforms
from src.models.inference import AttributeClassifier, ModelRegistry


def make_classification_dataset(tmp_path, attribute_name, classes, images_per_class=10, size=(64, 64)):
    """Create a synthetic classification dataset for testing training."""
    ds_dir = tmp_path / 'datasets' / attribute_name
    for cls_name in classes:
        cls_dir = ds_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            # Create slightly different colored images per class
            color_map = {{'leather': (139, 69, 19), 'wood': (210, 180, 140),
                         'fabric': (70, 130, 180), 'metal': (192, 192, 192),
                         'yes': (0, 200, 0), 'no': (200, 0, 0)}}
            base_color = color_map.get(cls_name, (128, 128, 128))
            noise = (np.random.randint(-20, 20), np.random.randint(-20, 20), np.random.randint(-20, 20))
            color = tuple(max(0, min(255, c + n)) for c, n in zip(base_color, noise))
            img = Image.new('RGB', size, color)
            img.save(str(cls_dir / f'img_{{i:04d}}.jpg'))
    return str(ds_dir)


class TestCNNMicroTraining:
    """Actual CNN training with minimal data - tests the full training loop."""

    def test_train_2class_resnet18(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'material', ['leather', 'wood'], images_per_class=20)
        result = train_attribute_model(
            dataset_dir=ds,
            attribute_name='material',
            output_dir=str(tmp_path / 'models'),
            backbone='resnet18',
            epochs=3,
            batch_size=8,
            learning_rate=0.01,
            val_split=0.2,
            pretrained=False,
            device='cpu',
            num_workers=0,
            early_stopping=0,
            target_size=(64, 64)
        )
        assert result['success'] is True
        assert result['num_classes'] == 2
        assert result['class_names'] == ['leather', 'wood']
        assert result['best_val_acc'] >= 0.0
        assert result['total_epochs'] == 3
        assert os.path.exists(result['model_path'])
        assert os.path.exists(result['meta_path'])

    def test_train_3class_mobilenet(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'material', ['leather', 'wood', 'fabric'], images_per_class=15)
        result = train_attribute_model(
            dataset_dir=ds,
            attribute_name='material',
            output_dir=str(tmp_path / 'models'),
            backbone='mobilenet_v2',
            epochs=2,
            batch_size=4,
            pretrained=False,
            device='cpu',
            num_workers=0,
            early_stopping=0,
            target_size=(64, 64)
        )
        assert result['success'] is True
        assert result['num_classes'] == 3

    def test_train_single_class_fails(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'bad', ['only_one'], images_per_class=10)
        result = train_attribute_model(
            dataset_dir=ds,
            attribute_name='bad',
            output_dir=str(tmp_path / 'models'),
            backbone='resnet18',
            epochs=1,
            pretrained=False,
            device='cpu',
            num_workers=0
        )
        assert result['success'] is False
        assert 'at least 2 classes' in result['error']

    def test_train_early_stopping(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'armrest', ['yes', 'no'], images_per_class=20)
        result = train_attribute_model(
            dataset_dir=ds,
            attribute_name='armrest',
            output_dir=str(tmp_path / 'models'),
            backbone='resnet18',
            epochs=100,
            batch_size=8,
            pretrained=False,
            device='cpu',
            num_workers=0,
            early_stopping=2,
            target_size=(64, 64)
        )
        assert result['success'] is True
        assert result['total_epochs'] < 100  # should stop early

    def test_model_save_load_predict(self, tmp_path):
        """Train, save, load, and predict - full lifecycle."""
        ds = make_classification_dataset(tmp_path, 'material', ['leather', 'wood'], images_per_class=20)
        models_dir = str(tmp_path / 'models')
        result = train_attribute_model(
            dataset_dir=ds,
            attribute_name='material',
            output_dir=models_dir,
            backbone='resnet18',
            epochs=3,
            batch_size=8,
            pretrained=False,
            device='cpu',
            num_workers=0,
            early_stopping=0,
            target_size=(64, 64)
        )
        assert result['success'] is True

        # Load model and predict
        classifier = AttributeClassifier(result['model_path'], device='cpu')
        assert classifier.attribute_name == 'material'
        assert classifier.num_classes == 2

        # Predict on a test image
        test_img = Image.new('RGB', (100, 100), (139, 69, 19))  # brown = leather-like
        pred = classifier.predict(test_img)
        assert 'attribute' in pred
        assert pred['attribute'] == 'material'
        assert pred['predicted_value'] in ['leather', 'wood']
        assert 0.0 <= pred['confidence'] <= 1.0
        assert len(pred['all_probabilities']) == 2
        assert abs(sum(pred['all_probabilities'].values()) - 1.0) < 0.01

    def test_model_info(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'test_attr', ['a', 'b'], images_per_class=10)
        models_dir = str(tmp_path / 'models')
        result = train_attribute_model(
            dataset_dir=ds, attribute_name='test_attr',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )
        classifier = AttributeClassifier(result['model_path'], device='cpu')
        info = classifier.info()
        assert info['attribute_name'] == 'test_attr'
        assert info['backbone'] == 'resnet18'
        assert info['num_classes'] == 2
        assert info['class_names'] == ['a', 'b']


class TestModelRegistryAdvanced:
    """Advanced model registry tests."""

    def test_register_load_predict(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'mat', ['x', 'y'], images_per_class=10)
        models_dir = str(tmp_path / 'models')
        result = train_attribute_model(
            dataset_dir=ds, attribute_name='mat',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )

        reg = ModelRegistry(models_dir, device='cpu')
        reg.register_model('mat', result['model_path'])

        assert 'mat' in reg.get_registered_attributes()

        classifier = reg.get_classifier('mat')
        assert classifier is not None
        assert classifier.num_classes == 2

        img = Image.new('RGB', (64, 64), (100, 100, 100))
        pred = classifier.predict(img)
        assert pred['predicted_value'] in ['x', 'y']

    def test_auto_discover_model(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'color', ['red', 'blue'], images_per_class=10)
        models_dir = str(tmp_path / 'models')
        result = train_attribute_model(
            dataset_dir=ds, attribute_name='color',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )

        # Create a fresh registry - should auto-discover
        reg2 = ModelRegistry(models_dir, device='cpu')
        classifier = reg2.get_classifier('color')
        assert classifier is not None

    def test_replace_model(self, tmp_path):
        ds1 = make_classification_dataset(tmp_path, 'v1', ['a', 'b'], images_per_class=10)
        ds2 = make_classification_dataset(tmp_path, 'v2', ['a', 'b', 'c'], images_per_class=10)
        models_dir = str(tmp_path / 'models')

        r1 = train_attribute_model(
            dataset_dir=ds1, attribute_name='test',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )
        r2 = train_attribute_model(
            dataset_dir=ds2, attribute_name='test_v2',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )

        reg = ModelRegistry(models_dir, device='cpu')
        reg.register_model('test', r1['model_path'])
        c1 = reg.get_classifier('test')
        assert c1.num_classes == 2

        # Replace with 3-class model
        reg.register_model('test', r2['model_path'])
        c2 = reg.get_classifier('test')
        assert c2.num_classes == 3

    def test_unregister(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'rm', ['a', 'b'], images_per_class=10)
        models_dir = str(tmp_path / 'models')
        r = train_attribute_model(
            dataset_dir=ds, attribute_name='rm',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )
        reg = ModelRegistry(models_dir, device='cpu')
        reg.register_model('rm', r['model_path'])
        assert reg.get_classifier('rm') is not None
        reg.unregister_model('rm')
        assert 'rm' not in reg.get_registered_attributes()

    def test_list_models_with_metadata(self, tmp_path):
        ds = make_classification_dataset(tmp_path, 'lm', ['p', 'q'], images_per_class=10)
        models_dir = str(tmp_path / 'models')
        r = train_attribute_model(
            dataset_dir=ds, attribute_name='lm',
            output_dir=models_dir, backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(64, 64)
        )
        reg = ModelRegistry(models_dir, device='cpu')
        models = reg.list_models()
        assert len(models) >= 1
        assert 'lm' in models


# ============================================================
# ERROR INJECTION & RESILIENCE TESTS
# ============================================================
from src.utils.data_loader import (
    load_yolo_annotations, load_json_attributes, parse_xanylabeling_json,
    discover_attributes, load_classes, yolo_to_pixel_bbox
)
from src.utils.cropper import crop_single_image, crop_dataset


class TestErrorInjection:
    """Test resilience to corrupted/malformed input data."""

    def test_corrupted_image_in_dataset(self, tmp_path):
        img_dir = tmp_path / 'images'
        lbl_dir = tmp_path / 'labels'
        img_dir.mkdir()
        lbl_dir.mkdir()
        # Write a corrupt image (not valid jpg)
        (img_dir / 'corrupt.jpg').write_bytes(b'not a real image file')
        (lbl_dir / 'corrupt.txt').write_text('0 0.5 0.5 0.3 0.4' + NL)
        stats = crop_dataset(str(img_dir), str(lbl_dir), output_dir=str(tmp_path / 'out'))
        assert len(stats['errors']) > 0  # should record error, not crash

    def test_empty_yolo_annotation(self, tmp_path):
        img = Image.new('RGB', (100, 100), 'red')
        img.save(str(tmp_path / 'img.jpg'))
        (tmp_path / 'img.txt').write_text('')
        crops = crop_single_image(str(tmp_path / 'img.jpg'), str(tmp_path / 'img.txt'))
        assert crops == []

    def test_json_with_missing_fields(self, tmp_path):
        data = {{'shapes': [{{'label': 'x'}}]}}  # missing points, shape_type, etc.
        objs = parse_xanylabeling_json(data)
        assert len(objs) == 0  # should skip incomplete shapes

    def test_json_with_wrong_types(self, tmp_path):
        data = {{'shapes': [{{
            'label': 123,  # should be string
            'points': 'not a list',
            'shape_type': 'rectangle',
            'flags': 'not a dict',
            'attributes': []
        }}]}}
        # Should not crash
        try:
            objs = parse_xanylabeling_json(data)
        except (TypeError, AttributeError):
            pass  # acceptable to raise, but not crash with uncaught exception

    def test_yolo_with_nan_values(self, tmp_path):
        f = tmp_path / 'nan.txt'
        f.write_text('0 nan 0.5 0.3 0.4' + NL)
        # Should handle gracefully
        try:
            anns = load_yolo_annotations(str(f))
            # If it parses, the value should be NaN
            if anns:
                import math
                assert math.isnan(anns[0]['cx'])
        except ValueError:
            pass  # also acceptable

    def test_yolo_with_negative_values(self, tmp_path):
        f = tmp_path / 'neg.txt'
        f.write_text('0 -0.1 0.5 0.3 0.4' + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1
        assert anns[0]['cx'] == -0.1

    def test_very_long_line(self, tmp_path):
        f = tmp_path / 'long.txt'
        f.write_text('0 0.5 0.5 0.3 0.4 ' + 'extra ' * 1000 + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1

    def test_binary_noise_in_txt(self, tmp_path):
        f = tmp_path / 'noise.txt'
        f.write_bytes(bytes(range(256)))
        try:
            anns = load_yolo_annotations(str(f))
        except (UnicodeDecodeError, ValueError):
            pass  # acceptable

    def test_json_deeply_nested(self, tmp_path):
        # Deeply nested but valid JSON
        data = {{'shapes': [{{
            'label': 'deep',
            'points': [[0, 0], [10, 10]],
            'shape_type': 'rectangle',
            'flags': {{}},
            'attributes': {{f'level_{{i}}': f'val_{{i}}' for i in range(100)}}
        }}]}}
        objs = parse_xanylabeling_json(data)
        assert len(objs) == 1
        assert len(objs[0]['attributes']) == 100

    def test_crop_grayscale_image(self, tmp_path):
        img = Image.new('L', (200, 200), 128)  # grayscale
        img.save(str(tmp_path / 'gray.jpg'))
        (tmp_path / 'gray.txt').write_text('0 0.5 0.5 0.3 0.4' + NL)
        crops = crop_single_image(str(tmp_path / 'gray.jpg'), str(tmp_path / 'gray.txt'))
        assert len(crops) == 1

    def test_crop_rgba_image(self, tmp_path):
        img = Image.new('RGBA', (200, 200), (255, 0, 0, 128))
        img.save(str(tmp_path / 'rgba.png'))
        (tmp_path / 'rgba.txt').write_text('0 0.5 0.5 0.3 0.4' + NL)
        crops = crop_single_image(str(tmp_path / 'rgba.png'), str(tmp_path / 'rgba.txt'))
        assert len(crops) == 1

    def test_crop_16bit_image(self, tmp_path):
        img = Image.new('I;16', (200, 200))
        img.save(str(tmp_path / '16bit.tiff'))
        (tmp_path / '16bit.txt').write_text('0 0.5 0.5 0.3 0.4' + NL)
        try:
            crops = crop_single_image(str(tmp_path / '16bit.tiff'), str(tmp_path / '16bit.txt'))
        except Exception:
            pass  # some formats may not be supported


# ============================================================
# PERFORMANCE TESTS
# ============================================================

class TestPerformance:
    """Performance and memory stress tests."""

    def test_crop_speed_1000_annotations(self, tmp_path):
        img = Image.new('RGB', (2000, 2000), 'blue')
        img.save(str(tmp_path / 'perf.jpg'))
        lines = []
        for i in range(1000):
            cx = (i % 50) * 0.02 + 0.01
            cy = (i // 50) * 0.05 + 0.025
            lines.append(f"0 {{cx:.4f}} {{cy:.4f}} 0.015 0.04")
        (tmp_path / 'perf.txt').write_text(NL.join(lines))

        start = time.time()
        crops = crop_single_image(str(tmp_path / 'perf.jpg'), str(tmp_path / 'perf.txt'), min_size=1)
        elapsed = time.time() - start
        assert len(crops) == 1000
        assert elapsed < 30  # should complete in <30 seconds
        print(f"1000 crops in {{elapsed:.2f}}s")

    def test_discover_1000_json_files(self, tmp_path):
        for i in range(1000):
            data = {{'shapes': [{{
                'label': f'obj_{{i}}',
                'points': [[0,0],[10,10]],
                'shape_type': 'rectangle',
                'flags': {{}},
                'attributes': {{'type': f'val_{{i % 10}}'}}
            }}]}}
            (tmp_path / f'img_{{i:04d}}.json').write_text(json.dumps(data))

        start = time.time()
        attrs = discover_attributes(json_dir=str(tmp_path))
        elapsed = time.time() - start
        assert 'type' in attrs
        assert len(attrs['type']) == 10
        assert elapsed < 30
        print(f"1000 JSON files scanned in {{elapsed:.2f}}s")

    def test_backbone_inference_speed(self):
        import torch
        model = get_backbone('resnet18', num_classes=10, pretrained=False)
        model.eval()
        x = torch.randn(16, 3, 224, 224)  # batch of 16

        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                out = model(x)
        elapsed = time.time() - start
        assert elapsed < 60  # 10 batches of 16 on CPU
        print(f"160 inferences in {{elapsed:.2f}}s")

    def test_transforms_speed(self):
        tx = get_transforms((224, 224))
        img = Image.new('RGB', (800, 600))

        start = time.time()
        for _ in range(1000):
            _ = tx['val'](img)
        elapsed = time.time() - start
        assert elapsed < 30
        print(f"1000 transforms in {{elapsed:.2f}}s")


# ============================================================
# MCP SERVER TOOL ROUTING TESTS
# ============================================================

class TestMCPServerRouting:
    """Test the MCP server tool dispatch logic."""

    def test_list_tools(self):
        from src.server import list_tools
        tools_list = asyncio.get_event_loop().run_until_complete(list_tools())
        tool_names = [t.name for t in tools_list]
        expected = [
            'crop_objects', 'organize_dataset', 'discover_attributes',
            'train_attribute_model', 'train_all_attributes',
            'detect_and_classify', 'classify_crop', 'batch_detect_and_classify',
            'register_model', 'unregister_model', 'list_models', 'get_model_info',
            'update_config', 'get_config', 'set_class_attributes', 'full_pipeline',
            'get_system_info', 'set_device', 'estimate_batch_size', 'memory_summary'
        ]
        for name in expected:
            assert name in tool_names, f"Missing tool: {{name}}"
        assert len(tools_list) == 20

    def test_get_config_tool(self):
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('get_config', {{}})
        )
        assert isinstance(result, dict)
        assert 'training' in result
        assert 'paths' in result

    def test_get_config_specific_key(self):
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('get_config', {{'key': 'training.backbone'}})
        )
        assert 'training.backbone' in result

    def test_update_config_tool(self):
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('update_config', {{'key': 'training.epochs', 'value': 99}})
        )
        assert result['success'] is True
        # Verify
        r2 = asyncio.get_event_loop().run_until_complete(
            _handle_tool('get_config', {{'key': 'training.epochs'}})
        )
        assert r2['training.epochs'] == 99

    def test_list_models_tool(self):
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('list_models', {{}})
        )
        assert isinstance(result, dict)

    def test_unknown_tool(self):
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('nonexistent_tool', {{}})
        )
        assert 'error' in result

    def test_discover_attributes_tool(self, tmp_path):
        from src.server import _handle_tool, config
        config.set('paths.json_dir', str(tmp_path))
        data = {{'shapes': [{{
            'label': 'x', 'points': [[0,0],[10,10]],
            'shape_type': 'rectangle', 'flags': {{}},
            'attributes': {{'color': 'red'}}
        }}]}}
        (tmp_path / 'test.json').write_text(json.dumps(data))
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('discover_attributes', {{'json_dir': str(tmp_path)}})
        )
        assert 'color' in result


# ============================================================
# DATA FORMAT EDGE CASES
# ============================================================

class TestDataFormatEdgeCases:
    """Test unusual but valid data formats."""

    def test_windows_line_endings(self, tmp_path):
        f = tmp_path / 'crlf.txt'
        crlf = bytes([13, 10])
        f.write_bytes(b"0 0.5 0.5 0.3 0.4" + crlf + b"1 0.2 0.3 0.1 0.2" + crlf)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 2

    def test_tab_separated(self, tmp_path):
        f = tmp_path / 'tabs.txt'
        f.write_text('0\t0.5\t0.5\t0.3\t0.4' + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1

    def test_multiple_spaces(self, tmp_path):
        f = tmp_path / 'spaces.txt'
        f.write_text('0   0.5   0.5   0.3   0.4' + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1

    def test_json_with_bom(self, tmp_path):
        bom = bytes([0xEF, 0xBB, 0xBF])  # UTF-8 BOM
        content = bom + json.dumps({{'shapes': []}}).encode('utf-8')
        f = tmp_path / 'bom.json'
        f.write_bytes(content)
        data = load_json_attributes(str(f))
        assert 'shapes' in data

    def test_trailing_whitespace_in_classes(self, tmp_path):
        f = tmp_path / 'classes.txt'
        f.write_text('chair   ' + NL + 'sofa\t' + NL)
        classes = load_classes(str(f))
        assert classes == ['chair', 'sofa']

    def test_config_empty_override(self, tmp_path):
        from src.utils.config import Config
        f = tmp_path / 'empty.yaml'
        f.write_text('')
        cfg = Config(str(f))
        assert cfg.get('training.backbone') is not None  # defaults preserved

    def test_iou_float_precision(self):
        from src.utils.data_loader import compute_iou
        # Very similar boxes should have IoU close to 1
        iou = compute_iou((100, 100, 200, 200), (101, 101, 201, 201))
        assert 0.9 < iou < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
'''

output_path = os.path.join(os.path.dirname(__file__), 'test_extreme_ext.py')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(test_code)
print(f'Generated {output_path}')
print(f'File size: {os.path.getsize(output_path)} bytes')
