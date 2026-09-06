"""Extreme tests for attribute-classifier-mcp.

Covers: data_loader, cropper, cnn_trainer, inference, config, server routing.
Tests edge cases, boundary conditions, malformed inputs, and stress scenarios.
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PIL import Image
import numpy as np

NL = chr(10)

# ============================================================
# DATA LOADER TESTS
# ============================================================
from src.utils.data_loader import (
    load_yolo_annotations,
    yolo_to_pixel_bbox,
    parse_xanylabeling_json,
    match_annotations_to_attributes,
    compute_iou,
    discover_attributes,
    load_classes,
    load_json_attributes
)


class TestLoadYoloAnnotations:
    """YOLO txt annotation parsing."""

    def test_normal_single_line(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("0 0.5 0.5 0.3 0.4" + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1
        assert anns[0] == {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.3, 'h': 0.4}

    def test_multiple_objects(self, tmp_path):
        f = tmp_path / "b.txt"
        f.write_text("0 0.1 0.2 0.3 0.4" + NL + "1 0.5 0.6 0.7 0.8" + NL + "2 0.9 0.1 0.2 0.3" + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 3
        assert anns[2]['class_id'] == 2

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 0

    def test_blank_lines(self, tmp_path):
        f = tmp_path / "blanks.txt"
        f.write_text(NL + NL + "0 0.5 0.5 0.3 0.4" + NL + NL + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1

    def test_malformed_line_short(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("0 0.5" + NL + "0 0.5 0.5 0.3 0.4" + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1

    def test_extra_columns(self, tmp_path):
        f = tmp_path / "extra.txt"
        f.write_text("0 0.5 0.5 0.3 0.4 0.9" + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1

    def test_large_class_id(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("999 0.5 0.5 0.3 0.4" + NL)
        anns = load_yolo_annotations(str(f))
        assert anns[0]['class_id'] == 999

    def test_boundary_values(self, tmp_path):
        f = tmp_path / "boundary.txt"
        f.write_text("0 0.0 0.0 1.0 1.0" + NL)
        anns = load_yolo_annotations(str(f))
        assert anns[0]['cx'] == 0.0
        assert anns[0]['w'] == 1.0

    def test_unicode_path(self, tmp_path):
        f = tmp_path / "中文标注.txt"
        f.write_text("0 0.5 0.5 0.3 0.4" + NL)
        anns = load_yolo_annotations(str(f))
        assert len(anns) == 1


class TestYoloToPixelBbox:
    """YOLO normalized to pixel bbox conversion."""

    def test_center_box(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.5, 'h': 0.5}
        bbox = yolo_to_pixel_bbox(ann, 100, 100)
        assert bbox == (25, 25, 75, 75)

    def test_full_image(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 1.0, 'h': 1.0}
        bbox = yolo_to_pixel_bbox(ann, 200, 100)
        assert bbox == (0, 0, 200, 100)

    def test_with_padding(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.5, 'h': 0.5}
        bbox = yolo_to_pixel_bbox(ann, 100, 100, padding=10)
        assert bbox == (15, 15, 85, 85)

    def test_padding_clamp(self):
        ann = {'class_id': 0, 'cx': 0.05, 'cy': 0.05, 'w': 0.1, 'h': 0.1}
        bbox = yolo_to_pixel_bbox(ann, 100, 100, padding=20)
        assert bbox[0] == 0
        assert bbox[1] == 0

    def test_tiny_image(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.5, 'h': 0.5}
        bbox = yolo_to_pixel_bbox(ann, 2, 2)
        assert all(isinstance(v, int) for v in bbox)

    def test_large_image(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.1, 'h': 0.1}
        bbox = yolo_to_pixel_bbox(ann, 10000, 10000)
        assert bbox == (4500, 4500, 5500, 5500)


class TestComputeIou:
    """IoU computation."""

    def test_identical(self):
        assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial(self):
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        expected = 25.0 / 175.0
        assert abs(iou - expected) < 1e-6

    def test_contained(self):
        iou = compute_iou((0, 0, 20, 20), (5, 5, 15, 15))
        assert abs(iou - (100.0 / 400.0)) < 1e-6

    def test_touching_edge(self):
        iou = compute_iou((0, 0, 10, 10), (10, 0, 20, 10))
        assert iou == 0.0

    def test_zero_area_box(self):
        iou = compute_iou((5, 5, 5, 5), (0, 0, 10, 10))
        assert iou == 0.0

    def test_negative_coords(self):
        iou = compute_iou((-10, -10, 0, 0), (-5, -5, 5, 5))
        assert iou > 0


class TestParseXAnyLabelingJson:
    """X-AnyLabeling JSON format parsing."""

    def test_basic(self):
        data = {
            'shapes': [{
                'label': 'chair',
                'points': [[10, 20], [100, 200]],
                'shape_type': 'rectangle',
                'flags': {'has_armrest': True},
                'attributes': {'material': 'leather'}
            }]
        }
        objs = parse_xanylabeling_json(data)
        assert len(objs) == 1
        assert objs[0]['label'] == 'chair'
        assert objs[0]['attributes']['material'] == 'leather'
        assert objs[0]['attributes']['has_armrest'] == 'yes'

    def test_bool_false_flag(self):
        data = {
            'shapes': [{
                'label': 'sofa',
                'points': [[0, 0], [50, 50]],
                'shape_type': 'rectangle',
                'flags': {'has_armrest': False},
                'attributes': {}
            }]
        }
        objs = parse_xanylabeling_json(data)
        assert objs[0]['attributes']['has_armrest'] == 'no'

    def test_empty_shapes(self):
        assert parse_xanylabeling_json({'shapes': []}) == []

    def test_no_shapes_key(self):
        assert parse_xanylabeling_json({}) == []

    def test_multiple_shapes(self):
        data = {'shapes': [
            {'label': 'a', 'points': [[0,0],[10,10]], 'shape_type': 'rectangle', 'flags': {}, 'attributes': {'x': '1'}},
            {'label': 'b', 'points': [[20,20],[30,30]], 'shape_type': 'rectangle', 'flags': {}, 'attributes': {'x': '2'}},
            {'label': 'c', 'points': [[40,40],[50,50]], 'shape_type': 'rectangle', 'flags': {}, 'attributes': {'x': '3'}},
        ]}
        objs = parse_xanylabeling_json(data)
        assert len(objs) == 3

    def test_polygon_shape(self):
        data = {'shapes': [{
            'label': 'table',
            'points': [[10, 10], [50, 10], [50, 50], [10, 50]],
            'shape_type': 'polygon',
            'flags': {},
            'attributes': {'material': 'wood'}
        }]}
        objs = parse_xanylabeling_json(data)
        assert objs[0]['bbox'] == (10, 10, 50, 50)

    def test_no_points(self):
        data = {'shapes': [{
            'label': 'empty',
            'points': [],
            'shape_type': 'rectangle',
            'flags': {},
            'attributes': {}
        }]}
        objs = parse_xanylabeling_json(data)
        assert len(objs) == 0

    def test_description_attributes(self):
        data = {'shapes': [{
            'label': 'chair',
            'points': [[0,0],[10,10]],
            'shape_type': 'rectangle',
            'flags': {},
            'attributes': {},
            'description': 'color=red,size=large'
        }]}
        objs = parse_xanylabeling_json(data)
        assert objs[0]['attributes']['color'] == 'red'
        assert objs[0]['attributes']['size'] == 'large'

    def test_attributes_override_flags(self):
        data = {'shapes': [{
            'label': 'x',
            'points': [[0,0],[10,10]],
            'shape_type': 'rectangle',
            'flags': {'material': 'flag_value'},
            'attributes': {'material': 'attr_value'}
        }]}
        objs = parse_xanylabeling_json(data)
        assert objs[0]['attributes']['material'] == 'attr_value'

    def test_none_attribute_value(self):
        data = {'shapes': [{
            'label': 'x',
            'points': [[0,0],[10,10]],
            'shape_type': 'rectangle',
            'flags': {},
            'attributes': {'color': None}
        }]}
        objs = parse_xanylabeling_json(data)
        assert objs[0]['attributes']['color'] == ''


class TestMatchAnnotations:
    """Match YOLO bbox to JSON attributes via IoU."""

    def test_perfect_match(self):
        yolo = [{'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.5, 'h': 0.5}]
        json_objs = [{'label': 'chair', 'bbox': (25, 25, 75, 75), 'attributes': {'material': 'wood'}, 'shape_type': 'rectangle'}]
        merged = match_annotations_to_attributes(yolo, json_objs, 100, 100)
        assert len(merged) == 1
        assert merged[0]['attributes']['material'] == 'wood'
        assert merged[0]['match_iou'] == 1.0

    def test_no_match_low_iou(self):
        yolo = [{'class_id': 0, 'cx': 0.1, 'cy': 0.1, 'w': 0.1, 'h': 0.1}]
        json_objs = [{'label': 'chair', 'bbox': (80, 80, 100, 100), 'attributes': {'m': 'x'}, 'shape_type': 'rectangle'}]
        merged = match_annotations_to_attributes(yolo, json_objs, 100, 100)
        assert merged[0]['attributes'] == {}

    def test_multiple_objects_greedy_match(self):
        yolo = [
            {'class_id': 0, 'cx': 0.25, 'cy': 0.25, 'w': 0.4, 'h': 0.4},
            {'class_id': 1, 'cx': 0.75, 'cy': 0.75, 'w': 0.4, 'h': 0.4},
        ]
        json_objs = [
            {'label': 'a', 'bbox': (5, 5, 45, 45), 'attributes': {'type': 'first'}, 'shape_type': 'rectangle'},
            {'label': 'b', 'bbox': (55, 55, 95, 95), 'attributes': {'type': 'second'}, 'shape_type': 'rectangle'},
        ]
        merged = match_annotations_to_attributes(yolo, json_objs, 100, 100)
        assert len(merged) == 2
        assert merged[0]['attributes']['type'] == 'first'
        assert merged[1]['attributes']['type'] == 'second'

    def test_more_yolo_than_json(self):
        yolo = [
            {'class_id': 0, 'cx': 0.25, 'cy': 0.25, 'w': 0.3, 'h': 0.3},
            {'class_id': 1, 'cx': 0.75, 'cy': 0.75, 'w': 0.3, 'h': 0.3},
        ]
        json_objs = [
            {'label': 'a', 'bbox': (10, 10, 40, 40), 'attributes': {'v': '1'}, 'shape_type': 'rectangle'},
        ]
        merged = match_annotations_to_attributes(yolo, json_objs, 100, 100)
        assert len(merged) == 2
        matched_count = sum(1 for m in merged if m['attributes'])
        assert matched_count == 1


class TestDiscoverAttributes:
    """Discover attributes from JSON annotations."""

    def test_from_json_dir(self, tmp_path):
        data = {'shapes': [
            {'label': 'chair', 'points': [[0,0],[10,10]], 'shape_type': 'rectangle',
             'flags': {}, 'attributes': {'material': 'leather', 'color': 'brown'}},
            {'label': 'sofa', 'points': [[0,0],[10,10]], 'shape_type': 'rectangle',
             'flags': {}, 'attributes': {'material': 'fabric', 'color': 'blue'}},
        ]}
        (tmp_path / 'img1.json').write_text(json.dumps(data))
        attrs = discover_attributes(json_dir=str(tmp_path))
        assert 'material' in attrs
        assert set(attrs['material']) == {'leather', 'fabric'}
        assert set(attrs['color']) == {'brown', 'blue'}

    def test_from_master_json(self, tmp_path):
        master = {
            'img1.jpg': {'objects': [{'attributes': {'material': 'wood', 'size': 'large'}}]},
            'img2.jpg': {'objects': [{'attributes': {'material': 'metal', 'size': 'small'}}]},
        }
        mf = tmp_path / 'master.json'
        mf.write_text(json.dumps(master))
        attrs = discover_attributes(master_json=str(mf))
        assert 'material' in attrs
        assert set(attrs['material']) == {'wood', 'metal'}

    def test_empty_dir(self, tmp_path):
        attrs = discover_attributes(json_dir=str(tmp_path))
        assert attrs == {}

    def test_malformed_json(self, tmp_path):
        (tmp_path / 'bad.json').write_text('{invalid json')
        attrs = discover_attributes(json_dir=str(tmp_path))
        assert attrs == {}

    def test_nonexistent_dir(self):
        attrs = discover_attributes(json_dir='/nonexistent/path')
        assert attrs == {}


class TestLoadClasses:
    """Load class names from txt."""

    def test_basic(self, tmp_path):
        f = tmp_path / 'classes.txt'
        f.write_text('chair' + NL + 'sofa' + NL + 'table' + NL)
        assert load_classes(str(f)) == ['chair', 'sofa', 'table']

    def test_blank_lines(self, tmp_path):
        f = tmp_path / 'classes.txt'
        f.write_text('chair' + NL + NL + 'sofa' + NL + NL)
        assert load_classes(str(f)) == ['chair', 'sofa']

    def test_whitespace(self, tmp_path):
        f = tmp_path / 'classes.txt'
        f.write_text('  chair  ' + NL + '  sofa' + NL)
        assert load_classes(str(f)) == ['chair', 'sofa']

    def test_unicode_classes(self, tmp_path):
        f = tmp_path / 'classes.txt'
        f.write_text('沙发' + NL + '桌子' + NL + '椅子' + NL)
        classes = load_classes(str(f))
        assert len(classes) == 3


# ============================================================
# CONFIG TESTS
# ============================================================
from src.utils.config import Config, get_config


class TestConfig:
    """Configuration manager."""

    def test_default_config_loads(self):
        cfg = Config()
        assert cfg.get('training.backbone') is not None

    def test_get_nested(self):
        cfg = Config()
        assert cfg.get('crop.padding') == 5
        assert cfg.get('crop.target_size') == [224, 224]

    def test_get_default(self):
        cfg = Config()
        assert cfg.get('nonexistent.key', 'fallback') == 'fallback'

    def test_set_and_get(self):
        cfg = Config()
        cfg.set('custom.key', 'value')
        assert cfg.get('custom.key') == 'value'

    def test_deep_set(self):
        cfg = Config()
        cfg.set('a.b.c.d', 42)
        assert cfg.get('a.b.c.d') == 42

    def test_save_and_reload(self, tmp_path):
        cfg = Config()
        cfg.set('test.value', 'hello')
        save_path = str(tmp_path / 'test_config.yaml')
        cfg.save(save_path)
        cfg2 = Config(save_path)
        assert cfg2.get('test.value') == 'hello'

    def test_override_merge(self, tmp_path):
        import yaml
        override = tmp_path / 'override.yaml'
        override.write_text(yaml.dump({'training': {'epochs': 999}}))
        cfg = Config(str(override))
        assert cfg.get('training.epochs') == 999
        assert cfg.get('training.backbone') is not None

    def test_to_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert 'training' in d


# ============================================================
# CROPPER TESTS
# ============================================================
from src.utils.cropper import crop_single_image, crop_dataset, organize_by_attribute


def make_test_image(path, size=(100, 100), color='red'):
    """Create a test image."""
    img = Image.new('RGB', size, color)
    img.save(path)
    return img


def make_test_dataset(tmp_path, num_images=3, num_objects=2):
    """Create a complete test dataset."""
    img_dir = tmp_path / 'images'
    lbl_dir = tmp_path / 'labels'
    json_dir = tmp_path / 'jsons'
    img_dir.mkdir()
    lbl_dir.mkdir()
    json_dir.mkdir()

    materials = ['leather', 'wood', 'fabric']
    armrests = ['yes', 'no']

    for i in range(num_images):
        img = Image.new('RGB', (200, 200), (i * 50, 100, 150))
        img.save(str(img_dir / f'img{i:03d}.jpg'))

        lines = []
        shapes = []
        for j in range(num_objects):
            cx = (j + 1) / (num_objects + 1)
            cy = 0.5
            w = 0.3
            h = 0.4
            lines.append(f"0 {cx} {cy} {w} {h}")
            x1 = int((cx - w/2) * 200)
            y1 = int((cy - h/2) * 200)
            x2 = int((cx + w/2) * 200)
            y2 = int((cy + h/2) * 200)
            shapes.append({
                'label': 'chair',
                'points': [[x1, y1], [x2, y2]],
                'shape_type': 'rectangle',
                'flags': {},
                'attributes': {
                    'material': materials[(i + j) % len(materials)],
                    'has_armrest': armrests[(i + j) % len(armrests)]
                }
            })
        (lbl_dir / f'img{i:03d}.txt').write_text(NL.join(lines) + NL)
        (json_dir / f'img{i:03d}.json').write_text(json.dumps({
            'shapes': shapes,
            'imagePath': f'img{i:03d}.jpg',
            'imageHeight': 200,
            'imageWidth': 200
        }))

    (tmp_path / 'classes.txt').write_text('chair' + NL)

    return img_dir, lbl_dir, json_dir


class TestCropSingleImage:
    """Crop objects from a single image."""

    def test_basic_crop(self, tmp_path):
        img_path = str(tmp_path / 'test.jpg')
        make_test_image(img_path, (200, 200))
        txt_path = str(tmp_path / 'test.txt')
        Path(txt_path).write_text('0 0.5 0.5 0.3 0.4' + NL)
        crops = crop_single_image(img_path, txt_path)
        assert len(crops) == 1
        assert crops[0]['class_id'] == 0
        assert crops[0]['crop_image'].size[0] > 0

    def test_with_json(self, tmp_path):
        img_path = str(tmp_path / 'test.jpg')
        make_test_image(img_path, (200, 200))
        txt_path = str(tmp_path / 'test.txt')
        Path(txt_path).write_text('0 0.5 0.5 0.3 0.4' + NL)
        json_path = str(tmp_path / 'test.json')
        Path(json_path).write_text(json.dumps({
            'shapes': [{
                'label': 'chair',
                'points': [[70, 60], [130, 140]],
                'shape_type': 'rectangle',
                'flags': {},
                'attributes': {'material': 'leather'}
            }]
        }))
        crops = crop_single_image(img_path, txt_path, json_path)
        assert len(crops) == 1
        assert crops[0]['attributes'].get('material') == 'leather'

    def test_multiple_objects(self, tmp_path):
        img_path = str(tmp_path / 'test.jpg')
        make_test_image(img_path, (400, 400))
        txt_path = str(tmp_path / 'test.txt')
        Path(txt_path).write_text('0 0.25 0.25 0.2 0.2' + NL + '1 0.75 0.75 0.2 0.2' + NL)
        crops = crop_single_image(img_path, txt_path)
        assert len(crops) == 2

    def test_tiny_bbox_filtered(self, tmp_path):
        img_path = str(tmp_path / 'test.jpg')
        make_test_image(img_path, (200, 200))
        txt_path = str(tmp_path / 'test.txt')
        Path(txt_path).write_text('0 0.5 0.5 0.01 0.01' + NL)
        crops = crop_single_image(img_path, txt_path, min_size=32)
        assert len(crops) == 0


class TestCropDataset:
    """Crop all objects from a dataset."""

    def test_basic_dataset(self, tmp_path):
        img_dir, lbl_dir, json_dir = make_test_dataset(tmp_path, 3, 2)
        out_dir = str(tmp_path / 'cropped')
        stats = crop_dataset(str(img_dir), str(lbl_dir), str(json_dir),
                            output_dir=out_dir, classes_file=str(tmp_path / 'classes.txt'))
        assert stats['total_images'] == 3
        assert stats['images_processed'] == 3
        assert stats['total_crops'] == 6
        assert os.path.isdir(out_dir)

    def test_missing_txt(self, tmp_path):
        img_dir = tmp_path / 'images'
        lbl_dir = tmp_path / 'labels'
        img_dir.mkdir()
        lbl_dir.mkdir()
        make_test_image(str(img_dir / 'img.jpg'))
        stats = crop_dataset(str(img_dir), str(lbl_dir), output_dir=str(tmp_path / 'out'))
        assert stats['images_processed'] == 0
        assert len(stats['errors']) > 0

    def test_empty_dirs(self, tmp_path):
        img_dir = tmp_path / 'images'
        lbl_dir = tmp_path / 'labels'
        img_dir.mkdir()
        lbl_dir.mkdir()
        stats = crop_dataset(str(img_dir), str(lbl_dir), output_dir=str(tmp_path / 'out'))
        assert stats['total_images'] == 0
        assert stats['total_crops'] == 0


class TestOrganizeByAttribute:
    """Organize cropped images by attribute value."""

    def test_basic_organize(self, tmp_path):
        cropped = tmp_path / 'cropped' / 'chair'
        cropped.mkdir(parents=True)

        for i, mat in enumerate(['leather', 'wood', 'leather', 'fabric']):
            make_test_image(str(cropped / f'img{i}_obj0.jpg'), (50, 50))
            (cropped / f'img{i}_obj0.json').write_text(json.dumps({
                'source_image': f'img{i}.jpg',
                'class_id': 0,
                'class_name': 'chair',
                'label': 'chair',
                'bbox': [10, 10, 50, 50],
                'attributes': {'material': mat, 'has_armrest': 'yes'}
            }))

        out_dir = str(tmp_path / 'datasets' / 'material')
        stats = organize_by_attribute(str(tmp_path / 'cropped'), attribute_name='material',
                                     output_dir=out_dir, target_size=(64, 64))
        assert stats['total_images'] == 4
        assert stats['values']['leather'] == 2
        assert stats['values']['wood'] == 1
        assert stats['values']['fabric'] == 1
        assert os.path.isdir(os.path.join(out_dir, 'leather'))


# ============================================================
# CNN TRAINER TESTS
# ============================================================
from src.models.cnn_trainer import get_backbone, get_transforms


class TestGetBackbone:
    """CNN backbone instantiation."""

    @pytest.mark.parametrize('name', ['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'])
    def test_all_backbones(self, name):
        model = get_backbone(name, num_classes=5, pretrained=False)
        assert model is not None
        import torch
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 5)

    def test_invalid_backbone(self):
        with pytest.raises(ValueError):
            get_backbone('invalid_model', 5)

    def test_single_class(self):
        model = get_backbone('resnet18', num_classes=1, pretrained=False)
        import torch
        out = model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 1)

    def test_many_classes(self):
        model = get_backbone('resnet18', num_classes=100, pretrained=False)
        import torch
        out = model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 100)


class TestGetTransforms:
    """Data augmentation transforms."""

    def test_train_transform(self):
        tx = get_transforms((224, 224), augment=True)
        assert 'train' in tx
        assert 'val' in tx

    def test_no_augment(self):
        tx = get_transforms((128, 128), augment=False)
        assert 'train' in tx

    def test_custom_size(self):
        tx = get_transforms((64, 64))
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        tensor = tx['val'](img)
        assert tensor.shape[1] == 64
        assert tensor.shape[2] == 64


# ============================================================
# MODEL REGISTRY TESTS
# ============================================================
from src.models.inference import ModelRegistry


class TestModelRegistry:
    """Model registration and lookup."""

    def test_empty_registry(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / 'models'))
        assert reg.list_models() == {}
        assert reg.get_registered_attributes() == []

    def test_register_nonexistent_model(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / 'models'))
        with pytest.raises(FileNotFoundError):
            reg.register_model('material', '/nonexistent/model.pth')

    def test_get_missing_classifier(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / 'models'))
        assert reg.get_classifier('nonexistent') is None


# ============================================================
# INTEGRATION: FULL PIPELINE SIMULATION
# ============================================================

class TestFullPipelineIntegration:
    """End-to-end integration test with synthetic data."""

    def test_crop_organize_flow(self, tmp_path):
        """Test: create data -> crop -> organize by attribute."""
        img_dir, lbl_dir, json_dir = make_test_dataset(tmp_path, 5, 3)

        cropped_dir = str(tmp_path / 'cropped')
        stats = crop_dataset(
            str(img_dir), str(lbl_dir), str(json_dir),
            output_dir=cropped_dir,
            classes_file=str(tmp_path / 'classes.txt')
        )
        assert stats['total_crops'] == 15

        mat_dir = str(tmp_path / 'datasets' / 'material')
        mat_stats = organize_by_attribute(
            cropped_dir, attribute_name='material',
            output_dir=mat_dir, target_size=(64, 64)
        )
        assert mat_stats['total_images'] == 15
        assert len(mat_stats['values']) == 3

        arm_dir = str(tmp_path / 'datasets' / 'has_armrest')
        arm_stats = organize_by_attribute(
            cropped_dir, attribute_name='has_armrest',
            output_dir=arm_dir, target_size=(64, 64)
        )
        assert arm_stats['total_images'] == 15
        assert set(arm_stats['values'].keys()) == {'yes', 'no'}

        for val in ['leather', 'wood', 'fabric']:
            assert os.path.isdir(os.path.join(mat_dir, val))
        for val in ['yes', 'no']:
            assert os.path.isdir(os.path.join(arm_dir, val))


# ============================================================
# STRESS TESTS
# ============================================================

class TestStress:
    """Boundary and stress scenarios."""

    def test_100_objects_per_image(self, tmp_path):
        img_path = str(tmp_path / 'stress.jpg')
        make_test_image(img_path, (1000, 1000))
        txt_path = str(tmp_path / 'stress.txt')
        lines = []
        for i in range(100):
            cx = (i % 10) * 0.1 + 0.05
            cy = (i // 10) * 0.1 + 0.05
            lines.append(f"0 {cx} {cy} 0.08 0.08")
        Path(txt_path).write_text(NL.join(lines))
        crops = crop_single_image(img_path, txt_path, min_size=10)
        assert len(crops) == 100

    def test_very_large_image(self, tmp_path):
        img_path = str(tmp_path / 'large.jpg')
        make_test_image(img_path, (4000, 3000))
        txt_path = str(tmp_path / 'large.txt')
        Path(txt_path).write_text('0 0.5 0.5 0.1 0.1' + NL)
        crops = crop_single_image(img_path, txt_path)
        assert len(crops) == 1
        assert crops[0]['crop_image'].size[0] > 0

    def test_1x1_image(self, tmp_path):
        img_path = str(tmp_path / 'tiny.jpg')
        make_test_image(img_path, (1, 1))
        txt_path = str(tmp_path / 'tiny.txt')
        Path(txt_path).write_text('0 0.5 0.5 1.0 1.0' + NL)
        crops = crop_single_image(img_path, txt_path, min_size=1)
        assert isinstance(crops, list)

    def test_discover_many_attributes(self, tmp_path):
        shapes = []
        for i in range(50):
            attrs = {f'attr_{j}': f'val_{i}_{j}' for j in range(20)}
            shapes.append({
                'label': f'obj_{i}',
                'points': [[0,0],[10,10]],
                'shape_type': 'rectangle',
                'flags': {},
                'attributes': attrs
            })
        (tmp_path / 'big.json').write_text(json.dumps({'shapes': shapes}))
        attrs = discover_attributes(json_dir=str(tmp_path))
        assert len(attrs) == 20
        for j in range(20):
            assert f'attr_{j}' in attrs
            assert len(attrs[f'attr_{j}']) == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
