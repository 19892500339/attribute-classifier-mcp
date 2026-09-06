"""Tests for adaptive training config, CPU/GPU modes, and robust file loading."""
import os
import sys
import gc
import json
import math
import time
import struct
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from PIL import Image

NL = chr(10)


# ==============================================================
# ADAPTIVE TRAINING CONFIG
# ==============================================================

class TestAdaptiveTrainingConfig:
    """Test auto-tune of training parameters."""

    def test_cpu_adaptive_default_60pct(self):
        """Default 60% resource limit on CPU."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        cfg = dm.adaptive_training_config(
            backbone='resnet18', num_classes=5, dataset_size=500
        )
        assert cfg['device'] == 'cpu'
        assert cfg['mode'] == 'cpu'
        assert cfg['resource_limit_percent'] == 60.0
        assert cfg['batch_size'] >= 1
        assert cfg['image_size'] == [224, 224]
        assert cfg['pin_memory'] is False
        assert cfg['mixed_precision'] is False
        assert 'resource_report' in cfg
        rr = cfg['resource_report']
        assert rr['estimated_usage_percent'] <= 60.0
        print(f"CPU adaptive: batch={cfg['batch_size']}, lr={cfg['learning_rate']}, "
              f"usage={rr['estimated_usage_percent']}%")

    def test_cpu_adaptive_30pct(self):
        """30% resource limit should give smaller batch."""
        from src.utils.device_manager import DeviceManager
        dm_60 = DeviceManager('cpu', max_resource_percent=60.0)
        dm_30 = DeviceManager('cpu', max_resource_percent=30.0)
        cfg_60 = dm_60.adaptive_training_config(backbone='resnet18', num_classes=5)
        cfg_30 = dm_30.adaptive_training_config(backbone='resnet18', num_classes=5)
        assert cfg_30['batch_size'] <= cfg_60['batch_size']
        print(f"60%: batch={cfg_60['batch_size']}, 30%: batch={cfg_30['batch_size']}")

    def test_cpu_adaptive_100pct(self):
        """100% should use more resources."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=100.0)
        cfg = dm.adaptive_training_config(backbone='resnet18', num_classes=10)
        assert cfg['batch_size'] >= 1
        assert cfg['resource_limit_percent'] == 100.0
        print(f"100% CPU: batch={cfg['batch_size']}")

    def test_adaptive_different_backbones(self):
        """Heavier backbones should get smaller batches."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        cfg_r18 = dm.adaptive_training_config(backbone='resnet18', num_classes=10)
        cfg_r50 = dm.adaptive_training_config(backbone='resnet50', num_classes=10)
        # resnet50 has ~2x params, should get <= batch of resnet18
        assert cfg_r50['batch_size'] <= cfg_r18['batch_size']
        print(f"resnet18: batch={cfg_r18['batch_size']}, resnet50: batch={cfg_r50['batch_size']}")

    def test_adaptive_different_image_sizes(self):
        """Larger images should get smaller batches."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        cfg_224 = dm.adaptive_training_config(backbone='resnet18', base_image_size=224)
        cfg_448 = dm.adaptive_training_config(backbone='resnet18', base_image_size=448)
        assert cfg_448['batch_size'] <= cfg_224['batch_size']
        print(f"224px: batch={cfg_224['batch_size']}, 448px: batch={cfg_448['batch_size']}")

    def test_adaptive_epoch_scaling(self):
        """More data should use fewer epochs."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        cfg_small = dm.adaptive_training_config(dataset_size=50)
        cfg_medium = dm.adaptive_training_config(dataset_size=500)
        cfg_large = dm.adaptive_training_config(dataset_size=5000)
        assert cfg_small['epochs'] >= cfg_medium['epochs'] >= cfg_large['epochs']
        print(f"Epochs: small={cfg_small['epochs']}, med={cfg_medium['epochs']}, large={cfg_large['epochs']}")

    def test_adaptive_accumulation_steps(self):
        """Small batches should compensate with accumulation."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        cfg = dm.adaptive_training_config(backbone='resnet18')
        assert cfg['effective_batch_size'] == cfg['batch_size'] * cfg['accumulation_steps']
        assert cfg['effective_batch_size'] >= cfg['batch_size']
        print(f"batch={cfg['batch_size']}, accum={cfg['accumulation_steps']}, "
              f"effective={cfg['effective_batch_size']}")

    def test_adaptive_resource_report_completeness(self):
        """Resource report should have all required fields."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        cfg = dm.adaptive_training_config()
        rr = cfg['resource_report']
        required = ['cpu_cores', 'ram_total_mb', 'ram_available_mb',
                    'ram_usable_mb', 'model_memory_mb', 'estimated_usage_mb',
                    'estimated_usage_percent', 'headroom_mb']
        for key in required:
            assert key in rr, f"Missing key: {key}"
        assert rr['headroom_mb'] >= 0, f"Negative headroom: {rr['headroom_mb']}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_adaptive_config(self):
        """GPU adaptive config should use VRAM."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cuda', max_resource_percent=60.0)
        cfg = dm.adaptive_training_config(backbone='resnet18', num_classes=10)
        assert cfg['device'].startswith('cuda')
        assert cfg['mode'] == 'gpu'
        assert cfg['pin_memory'] is True
        rr = cfg['resource_report']
        assert 'gpu_total_mb' in rr
        assert rr['estimated_usage_percent'] <= 60.0


# ==============================================================
# DEDICATED CPU/GPU MODE
# ==============================================================

class TestDedicatedModes:
    """Test CPU-dedicated and GPU-dedicated modes."""

    def test_cpu_dedicated_mode(self):
        """Apply CPU dedicated mode."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        result = dm.apply_cpu_dedicated_mode()
        assert result['mode'] == 'cpu_dedicated'
        assert result['device'] == 'cpu'
        assert result['resource_limit_percent'] == 60.0
        assert result['torch_threads'] >= 1
        # Torch threads should be ~60% of cores
        import os
        expected = max(1, int((os.cpu_count() or 1) * 0.6))
        assert result['torch_threads'] == expected
        print(f"CPU dedicated: threads={result['torch_threads']}")

    def test_cpu_dedicated_100pct(self):
        """100% should use all cores."""
        from src.utils.device_manager import DeviceManager
        import os
        dm = DeviceManager('cpu', max_resource_percent=100.0)
        result = dm.apply_cpu_dedicated_mode()
        assert result['torch_threads'] == (os.cpu_count() or 1)

    def test_gpu_dedicated_no_cuda(self):
        """GPU mode without CUDA should fallback gracefully."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)  # forced cpu
        result = dm.apply_gpu_dedicated_mode()
        assert 'error' in result
        assert result['device'] == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_dedicated_mode(self):
        """GPU dedicated mode should activate cudnn."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cuda', max_resource_percent=60.0)
        result = dm.apply_gpu_dedicated_mode()
        assert result['mode'] == 'gpu_dedicated'
        assert 'cuda' in result['device']
        assert result['cudnn_benchmark'] is True

    def test_set_device_percent_clamp(self):
        """Percent should be clamped 1-100."""
        from src.utils.device_manager import DeviceManager
        dm_low = DeviceManager('cpu', max_resource_percent=-10)
        assert dm_low.max_resource_percent == 1.0
        dm_high = DeviceManager('cpu', max_resource_percent=999)
        assert dm_high.max_resource_percent == 100.0

    def test_memory_summary_format(self):
        """Memory summary should be a non-empty string with device info."""
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager('cpu', max_resource_percent=60.0)
        summary = dm.memory_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'cpu' in summary.lower() or 'Device' in summary
        print(summary)


# ==============================================================
# ROBUST IMAGE FOLDER
# ==============================================================

class TestRobustImageFolder:
    """Test RobustImageFolder that skips bad files."""

    def test_skips_corrupt_image(self, tmp_path):
        """Corrupt .jpg should be skipped, not crash."""
        from src.models.cnn_trainer import RobustImageFolder
        # Create valid class directory with some good and bad images
        cls_a = tmp_path / 'a'
        cls_b = tmp_path / 'b'
        cls_a.mkdir(); cls_b.mkdir()
        # Good images
        for i in range(5):
            Image.new('RGB', (32, 32), (i*50, 100, 50)).save(str(cls_a / f'good{i}.jpg'))
            Image.new('RGB', (32, 32), (100, i*50, 50)).save(str(cls_b / f'good{i}.jpg'))
        # Corrupt image (random bytes)
        (cls_a / 'corrupt.jpg').write_bytes(b'not a real jpeg file at all')
        (cls_b / 'bad.png').write_bytes(bytes(range(256)))

        ds = RobustImageFolder(str(tmp_path))
        assert len(ds.skipped_files) == 2
        assert len(ds) == 10  # only good images
        # Should be able to iterate without error
        for i in range(len(ds)):
            img, label = ds[i]
            assert img is not None

    def test_skips_non_image_extensions(self, tmp_path):
        """Non-image files (.txt, .json) are ignored by ImageFolder loader.
        RobustImageFolder only sees image-like extensions."""
        from src.models.cnn_trainer import RobustImageFolder
        cls_a = tmp_path / 'a'
        cls_b = tmp_path / 'b'
        cls_a.mkdir(); cls_b.mkdir()
        Image.new('RGB', (32, 32)).save(str(cls_a / 'good.jpg'))
        Image.new('RGB', (32, 32)).save(str(cls_b / 'good.jpg'))
        # Non-image files - ImageFolder ignores these entirely
        (cls_a / 'readme.txt').write_text('hello')
        (cls_a / 'data.json').write_text('{}')
        (cls_b / 'notes.md').write_text('# notes')

        ds = RobustImageFolder(str(tmp_path))
        # Only the 2 valid images should be loaded, non-image files ignored
        assert len(ds) == 2

    def test_handles_truncated_image(self, tmp_path):
        """Truncated JPEG should be skipped."""
        from src.models.cnn_trainer import RobustImageFolder
        cls_a = tmp_path / 'a'
        cls_b = tmp_path / 'b'
        cls_a.mkdir(); cls_b.mkdir()
        # Create a real JPEG then truncate it
        good_path = cls_a / 'good.jpg'
        Image.new('RGB', (64, 64), 'red').save(str(good_path))
        data = good_path.read_bytes()
        truncated_path = cls_a / 'truncated.jpg'
        truncated_path.write_bytes(data[:len(data)//3])  # chop off 2/3
        Image.new('RGB', (64, 64)).save(str(cls_b / 'ok.jpg'))

        ds = RobustImageFolder(str(tmp_path))
        assert len(ds.skipped_files) >= 1
        # Should still be iterable
        for i in range(len(ds)):
            img, _ = ds[i]
            assert img is not None

    def test_converts_non_rgb(self, tmp_path):
        """Grayscale and RGBA images should be converted to RGB."""
        from src.models.cnn_trainer import RobustImageFolder
        from torchvision import transforms
        cls_a = tmp_path / 'a'
        cls_b = tmp_path / 'b'
        cls_a.mkdir(); cls_b.mkdir()
        Image.new('L', (32, 32), 128).save(str(cls_a / 'gray.png'))
        Image.new('RGBA', (32, 32), (255, 0, 0, 128)).save(str(cls_b / 'rgba.png'))

        tx = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = RobustImageFolder(str(tmp_path), transform=tx)
        assert len(ds) == 2
        for i in range(len(ds)):
            img, _ = ds[i]
            assert img.shape[0] == 3  # RGB channels

    def test_all_corrupt_dataset(self, tmp_path):
        """Dataset with only corrupt images should return 0 samples."""
        from src.models.cnn_trainer import RobustImageFolder
        cls_a = tmp_path / 'a'
        cls_b = tmp_path / 'b'
        cls_a.mkdir(); cls_b.mkdir()
        (cls_a / 'bad1.jpg').write_bytes(b'corrupt')
        (cls_b / 'bad2.jpg').write_bytes(b'nope')

        ds = RobustImageFolder(str(tmp_path))
        assert len(ds) == 0
        assert len(ds.skipped_files) == 2


# ==============================================================
# ROBUST DATA LOADING (YOLO + JSON + Cropper)
# ==============================================================

class TestRobustDataLoading:
    """Test that bad files are skipped during data loading."""

    def test_yolo_skips_nan_lines(self, tmp_path):
        """YOLO loader should skip NaN/Inf lines."""
        from src.utils.data_loader import load_yolo_annotations
        content = NL.join([
            '0 0.5 0.5 0.1 0.1',      # good
            '0 nan 0.5 0.1 0.1',       # NaN
            '0 0.5 inf 0.1 0.1',       # Inf
            '0 0.3 0.3 0.2 0.2',       # good
            'not a number at all',      # garbage
            '1 0.7 0.7 0.1 0.1',       # good
        ])
        txt = tmp_path / 'test.txt'
        txt.write_text(content)
        anns = load_yolo_annotations(str(txt))
        assert len(anns) == 3  # only 3 good lines

    def test_yolo_skips_short_lines(self, tmp_path):
        """Lines with fewer than 5 values should be skipped."""
        from src.utils.data_loader import load_yolo_annotations
        content = NL.join([
            '0 0.5 0.5 0.1 0.1',  # good
            '0 0.5',              # too short
            '0 0.5 0.5',          # too short
            '',                   # empty
            '1 0.2 0.2 0.3 0.3',  # good
        ])
        txt = tmp_path / 'test.txt'
        txt.write_text(content)
        anns = load_yolo_annotations(str(txt))
        assert len(anns) == 2

    def test_crop_skips_unreadable_image(self, tmp_path):
        """Cropper should return empty list for corrupt image."""
        from src.utils.cropper import crop_single_image
        (tmp_path / 'bad.jpg').write_bytes(b'not an image')
        (tmp_path / 'bad.txt').write_text('0 0.5 0.5 0.1 0.1')
        crops = crop_single_image(str(tmp_path / 'bad.jpg'), str(tmp_path / 'bad.txt'))
        assert crops == []

    def test_crop_skips_corrupt_json(self, tmp_path):
        """Cropper should still work if JSON is corrupt."""
        from src.utils.cropper import crop_single_image
        img = Image.new('RGB', (100, 100), 'red')
        img.save(str(tmp_path / 'test.jpg'))
        (tmp_path / 'test.txt').write_text('0 0.5 0.5 0.3 0.3')
        (tmp_path / 'test.json').write_text('{{not valid json')
        crops = crop_single_image(
            str(tmp_path / 'test.jpg'),
            str(tmp_path / 'test.txt'),
            str(tmp_path / 'test.json')
        )
        assert len(crops) == 1  # should work with YOLO only

    def test_crop_dataset_skips_bad_images(self, tmp_path):
        """crop_dataset should skip bad images and continue."""
        from src.utils.cropper import crop_dataset
        img_dir = tmp_path / 'images'
        lbl_dir = tmp_path / 'labels'
        img_dir.mkdir(); lbl_dir.mkdir()
        # 3 good images
        for i in range(3):
            Image.new('RGB', (100, 100), (i*80, 50, 100)).save(str(img_dir / f'img{i}.jpg'))
            (lbl_dir / f'img{i}.txt').write_text('0 0.5 0.5 0.3 0.3')
        # 1 corrupt image
        (img_dir / 'bad.jpg').write_bytes(b'corruption')
        (lbl_dir / 'bad.txt').write_text('0 0.5 0.5 0.3 0.3')

        stats = crop_dataset(str(img_dir), str(lbl_dir), output_dir=str(tmp_path / 'out'))
        # All 4 images are "processed" (no exception), but bad one yields 0 crops
        assert stats['images_processed'] == 4
        assert stats['total_crops'] == 3  # only 3 good images produce crops

    def test_discover_attributes_skips_bad_json(self, tmp_path):
        """discover_attributes should skip malformed JSON files."""
        from src.utils.data_loader import discover_attributes
        # 2 good JSONs
        for i in range(2):
            (tmp_path / f'good{i}.json').write_text(json.dumps({
                'shapes': [{
                    'label': 'chair',
                    'points': [[0, 0], [100, 100]],
                    'shape_type': 'rectangle',
                    'flags': {},
                    'attributes': {'material': 'wood'}
                }]
            }))
        # 1 corrupt JSON
        (tmp_path / 'bad.json').write_text('not valid json at all{')
        # 1 empty JSON
        (tmp_path / 'empty.json').write_text('{}')

        attrs = discover_attributes(json_dir=str(tmp_path))
        assert 'material' in attrs
        assert 'wood' in attrs['material']


# ==============================================================
# TRAINING WITH ADAPTIVE + ROBUST LOADING INTEGRATION
# ==============================================================

class TestAdaptiveTrainingIntegration:
    """Integration tests: adaptive config + robust loading in actual training."""

    def test_train_with_mixed_good_bad_files(self, tmp_path):
        """Training should succeed even if some images are corrupt."""
        from src.models.cnn_trainer import train_attribute_model
        ds = tmp_path / 'ds'
        for cls in ['a', 'b']:
            d = ds / cls
            d.mkdir(parents=True)
            for i in range(15):
                Image.new('RGB', (32, 32), (i*15, 100, 50)).save(str(d / f'good{i}.jpg'))
            # Add some bad files
            (d / 'corrupt.jpg').write_bytes(b'not a jpeg')
            (d / 'notes.txt').write_text('ignore me')

        result = train_attribute_model(
            dataset_dir=str(ds), attribute_name='mixed',
            output_dir=str(tmp_path / 'models'), backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(32, 32),
            batch_size=8
        )
        assert result['success'] is True
        assert result['train_samples'] + result['val_samples'] == 30
        print(f"Trained with {result['train_samples'] + result['val_samples']} valid images")

    def test_train_all_bad_returns_error(self, tmp_path):
        """If all images are bad, training should return success=False."""
        from src.models.cnn_trainer import train_attribute_model
        ds = tmp_path / 'ds'
        for cls in ['a', 'b']:
            d = ds / cls
            d.mkdir(parents=True)
            (d / 'bad1.jpg').write_bytes(b'corrupt')
            (d / 'bad2.jpg').write_bytes(b'also bad')

        result = train_attribute_model(
            dataset_dir=str(ds), attribute_name='allbad',
            output_dir=str(tmp_path / 'models'), backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(32, 32),
            batch_size=4
        )
        assert result['success'] is False

    def test_adaptive_config_used_in_training(self, tmp_path):
        """Verify adaptive config is applied when using defaults."""
        from src.models.cnn_trainer import train_attribute_model
        ds = tmp_path / 'ds'
        for cls in ['a', 'b']:
            d = ds / cls
            d.mkdir(parents=True)
            for i in range(10):
                Image.new('RGB', (32, 32)).save(str(d / f'img{i}.jpg'))

        result = train_attribute_model(
            dataset_dir=str(ds), attribute_name='adaptive_test',
            output_dir=str(tmp_path / 'models'), backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(32, 32)
            # batch_size and learning_rate NOT specified (use defaults >> adaptive)
        )
        assert result['success'] is True


# ==============================================================
# MCP SERVER TOOLS
# ==============================================================

class TestMCPAdaptiveTools:
    """Test MCP server adaptive/device tools."""

    def test_adaptive_training_config_tool(self):
        """MCP adaptive_training_config tool should return config."""
        import asyncio
        from src.server import _handle_tool
        config_result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('adaptive_training_config', {
                'backbone': 'resnet18',
                'num_classes': 5,
                'dataset_size': 500,
                'max_resource_percent': 60
            })
        )
        assert 'batch_size' in config_result
        assert 'learning_rate' in config_result
        assert 'resource_report' in config_result
        assert config_result['resource_limit_percent'] == 60.0

    def test_set_device_with_percent(self):
        """set_device tool should accept max_resource_percent."""
        import asyncio
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('set_device', {
                'device': 'cpu',
                'max_resource_percent': 40
            })
        )
        assert result['success'] is True
        assert result['device'] == 'cpu'
        assert result['is_cpu'] is True
        assert 'mode' in result

    def test_memory_summary_tool(self):
        """memory_summary tool should return summary text."""
        import asyncio
        from src.server import _handle_tool
        result = asyncio.get_event_loop().run_until_complete(
            _handle_tool('memory_summary', {})
        )
        assert 'summary' in result
        assert len(result['summary']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
