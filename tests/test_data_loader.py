"""Tests for data_loader module."""
import os
import json
import tempfile
import pytest

from src.utils.data_loader import (
    load_yolo_annotations,
    yolo_to_pixel_bbox,
    parse_xanylabeling_json,
    match_annotations_to_attributes,
    compute_iou,
    discover_attributes,
    load_classes
)


class TestLoadYoloAnnotations:
    def test_basic_parse(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.8 0.1 0.2\n")
        anns = load_yolo_annotations(str(txt))
        assert len(anns) == 2
        assert anns[0]['class_id'] == 0
        assert anns[0]['cx'] == 0.5
        assert anns[1]['class_id'] == 1

    def test_empty_file(self, tmp_path):
        txt = tmp_path / "empty.txt"
        txt.write_text("")
        anns = load_yolo_annotations(str(txt))
        assert len(anns) == 0


class TestYoloToPixelBbox:
    def test_center_box(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.5, 'h': 0.5}
        x1, y1, x2, y2 = yolo_to_pixel_bbox(ann, 100, 100)
        assert x1 == 25
        assert y1 == 25
        assert x2 == 75
        assert y2 == 75

    def test_with_padding(self):
        ann = {'class_id': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.5, 'h': 0.5}
        x1, y1, x2, y2 = yolo_to_pixel_bbox(ann, 100, 100, padding=10)
        assert x1 == 15
        assert y1 == 15
        assert x2 == 85
        assert y2 == 85


class TestComputeIou:
    def test_identical_boxes(self):
        iou = compute_iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert iou == 1.0

    def test_no_overlap(self):
        iou = compute_iou((0, 0, 10, 10), (20, 20, 30, 30))
        assert iou == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2  # intersection=25, union=175


class TestParseXAnyLabelingJson:
    def test_basic_shapes(self):
        data = {
            'shapes': [
                {
                    'label': 'chair',
                    'points': [[10, 20], [100, 200]],
                    'shape_type': 'rectangle',
                    'flags': {'has_armrest': True},
                    'attributes': {'material': 'leather'}
                }
            ]
        }
        objects = parse_xanylabeling_json(data)
        assert len(objects) == 1
        assert objects[0]['label'] == 'chair'
        assert objects[0]['attributes']['material'] == 'leather'
        assert objects[0]['attributes']['has_armrest'] == 'yes'


class TestDiscoverAttributes:
    def test_from_json_dir(self, tmp_path):
        json_data = {
            'shapes': [
                {
                    'label': 'chair',
                    'points': [[0, 0], [10, 10]],
                    'shape_type': 'rectangle',
                    'flags': {},
                    'attributes': {'material': 'leather', 'color': 'brown'}
                },
                {
                    'label': 'sofa',
                    'points': [[0, 0], [10, 10]],
                    'shape_type': 'rectangle',
                    'flags': {},
                    'attributes': {'material': 'fabric', 'color': 'blue'}
                }
            ]
        }
        jfile = tmp_path / 'test.json'
        jfile.write_text(json.dumps(json_data))

        attrs = discover_attributes(json_dir=str(tmp_path))
        assert 'material' in attrs
        assert 'leather' in attrs['material']
        assert 'fabric' in attrs['material']
        assert 'color' in attrs


class TestLoadClasses:
    def test_basic(self, tmp_path):
        cls_file = tmp_path / 'classes.txt'
        cls_file.write_text('chair\nsofa\ntable\n')
        classes = load_classes(str(cls_file))
        assert classes == ['chair', 'sofa', 'table']
