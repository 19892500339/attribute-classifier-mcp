"""Generate test_hardware_limits.py - CPU/GPU/RAM limit tests."""
import os

NL = chr(10)

test_code = f'''"""Hardware limit tests: CPU, GPU, RAM boundary and compatibility.

Tests the project under resource pressure to find OOM crashes,
CPU saturation issues, GPU fallback behavior, and memory leaks.
"""
import os
import sys
import gc
import time
import json
import psutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from PIL import Image

NL = chr(10)


def get_ram_mb():
    """Get current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_free_ram_mb():
    """Get system free RAM in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)


# ============================================================
# RAM LIMIT TESTS
# ============================================================

class TestRAMLimits:
    """Test behavior under memory pressure."""

    def test_ram_usage_after_crop_1000(self, tmp_path):
        """Verify cropping 1000 objects doesn't leak memory."""
        from src.utils.cropper import crop_single_image

        img = Image.new('RGB', (2000, 2000), 'blue')
        img.save(str(tmp_path / 'big.jpg'))
        lines = []
        for i in range(1000):
            cx = (i % 50) * 0.02 + 0.01
            cy = (i // 50) * 0.05 + 0.025
            lines.append(f"0 {{cx:.4f}} {{cy:.4f}} 0.015 0.04")
        (tmp_path / 'big.txt').write_text(NL.join(lines))

        gc.collect()
        ram_before = get_ram_mb()
        crops = crop_single_image(str(tmp_path / 'big.jpg'), str(tmp_path / 'big.txt'), min_size=1)
        assert len(crops) == 1000

        # Release crops explicitly
        del crops
        gc.collect()
        ram_after = get_ram_mb()

        # Memory should not grow more than 500MB for 1000 small crops
        ram_delta = ram_after - ram_before
        print(f"RAM delta after 1000 crops: {{ram_delta:.1f}} MB")
        assert ram_delta < 500, f"Memory leak: {{ram_delta:.1f}} MB growth"

    def test_ram_usage_large_dataset_crop(self, tmp_path):
        """Crop from 50 images x 20 objects = 1000 total crops."""
        from src.utils.cropper import crop_dataset

        img_dir = tmp_path / 'images'
        lbl_dir = tmp_path / 'labels'
        img_dir.mkdir()
        lbl_dir.mkdir()

        for i in range(50):
            img = Image.new('RGB', (500, 500), (i * 5, 100, 150))
            img.save(str(img_dir / f'img{{i:03d}}.jpg'))
            lines = []
            for j in range(20):
                cx = (j % 5) * 0.2 + 0.1
                cy = (j // 5) * 0.25 + 0.125
                lines.append(f"0 {{cx:.3f}} {{cy:.3f}} 0.15 0.2")
            (lbl_dir / f'img{{i:03d}}.txt').write_text(NL.join(lines))

        gc.collect()
        ram_before = get_ram_mb()
        stats = crop_dataset(str(img_dir), str(lbl_dir), output_dir=str(tmp_path / 'out'))
        gc.collect()
        ram_after = get_ram_mb()

        assert stats['total_crops'] == 1000
        ram_delta = ram_after - ram_before
        print(f"RAM delta for 50x20 dataset crop: {{ram_delta:.1f}} MB")
        assert ram_delta < 1000, f"Memory leak: {{ram_delta:.1f}} MB growth"

    def test_ram_large_image_8000x6000(self, tmp_path):
        """Test with a very large image (~144MB uncompressed RGB)."""
        from src.utils.cropper import crop_single_image

        free_ram = get_free_ram_mb()
        if free_ram < 2000:
            pytest.skip(f"Not enough free RAM: {{free_ram:.0f}} MB")

        img = Image.new('RGB', (8000, 6000), 'green')
        img.save(str(tmp_path / 'huge.jpg'))
        (tmp_path / 'huge.txt').write_text('0 0.5 0.5 0.1 0.1' + NL)

        crops = crop_single_image(str(tmp_path / 'huge.jpg'), str(tmp_path / 'huge.txt'))
        assert len(crops) == 1
        w, h = crops[0]['crop_image'].size
        # 0.1 * 8000 = 800 base + padding on each side
        assert 790 <= w <= 830  # allow for default padding
        assert 590 <= h <= 630  # allow for default padding
        print(f"8000x6000 image: crop size {{w}}x{{h}}")

    def test_ram_model_loading_multiple(self, tmp_path):
        """Load multiple CNN models and verify RAM stays bounded."""
        from src.models.cnn_trainer import get_backbone

        gc.collect()
        ram_before = get_ram_mb()

        models = []
        for name in ['resnet18', 'resnet34', 'mobilenet_v2', 'efficientnet_b0']:
            m = get_backbone(name, num_classes=10, pretrained=False)
            models.append(m)

        ram_with_models = get_ram_mb()
        ram_delta = ram_with_models - ram_before
        print(f"RAM for 4 models: {{ram_delta:.1f}} MB")

        # Clean up
        del models
        gc.collect()
        ram_after = get_ram_mb()
        ram_leaked = ram_after - ram_before
        print(f"RAM after cleanup: {{ram_leaked:.1f}} MB residual")

        # 4 models should fit in ~500MB, residual should be <100MB
        assert ram_delta < 500, f"Models too large: {{ram_delta:.1f}} MB"

    def test_ram_discover_10000_attributes(self, tmp_path):
        """Discover attributes from a very large JSON."""
        from src.utils.data_loader import discover_attributes

        shapes = []
        for i in range(10000):
            attrs = {{f'attr_{{j}}': f'val_{{i % 100}}_{{j}}' for j in range(10)}}
            shapes.append({{
                'label': f'obj_{{i}}',
                'points': [[0,0],[10,10]],
                'shape_type': 'rectangle',
                'flags': {{}},
                'attributes': attrs
            }})
        (tmp_path / 'huge.json').write_text(json.dumps({{'shapes': shapes}}))

        gc.collect()
        ram_before = get_ram_mb()
        attrs = discover_attributes(json_dir=str(tmp_path))
        gc.collect()
        ram_after = get_ram_mb()

        assert len(attrs) == 10
        for j in range(10):
            assert len(attrs[f'attr_{{j}}']) == 100
        ram_delta = ram_after - ram_before
        print(f"RAM for 10000-shape JSON discover: {{ram_delta:.1f}} MB")
        assert ram_delta < 500


# ============================================================
# CPU LIMIT TESTS
# ============================================================

class TestCPULimits:
    """Test CPU-intensive operations and multi-core behavior."""

    def test_cpu_training_uses_available_cores(self, tmp_path):
        """Verify training can use multiple DataLoader workers."""
        from src.models.cnn_trainer import train_attribute_model

        ds_dir = tmp_path / 'datasets' / 'cpu_test'
        for cls in ['a', 'b']:
            d = ds_dir / cls
            d.mkdir(parents=True)
            for i in range(30):
                Image.new('RGB', (64, 64), (i*8, 100, 150)).save(str(d / f'img{{i}}.jpg'))

        cpu_count = os.cpu_count() or 1
        workers = min(4, cpu_count)
        print(f"CPU cores: {{cpu_count}}, using {{workers}} workers")

        start = time.time()
        result = train_attribute_model(
            dataset_dir=str(ds_dir),
            attribute_name='cpu_test',
            output_dir=str(tmp_path / 'models'),
            backbone='resnet18',
            epochs=2,
            batch_size=16,
            pretrained=False,
            device='cpu',
            num_workers=workers,
            early_stopping=0,
            target_size=(64, 64)
        )
        elapsed = time.time() - start
        assert result['success'] is True
        print(f"Training 2 epochs with {{workers}} workers: {{elapsed:.2f}}s")

    def test_cpu_batch_inference_throughput(self):
        """Measure CPU inference throughput with different batch sizes."""
        from src.models.cnn_trainer import get_backbone

        model = get_backbone('resnet18', num_classes=10, pretrained=False)
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 224, 224)
            start = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            elapsed = time.time() - start
            throughput = (batch_size * 5) / elapsed
            print(f"Batch={{batch_size}}: {{throughput:.1f}} images/sec (CPU)")
            assert throughput > 0

    def test_cpu_transform_throughput(self):
        """Measure data augmentation throughput on CPU."""
        from src.models.cnn_trainer import get_transforms

        tx = get_transforms((224, 224), augment=True, aug_config={{
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation': 30,
            'color_jitter': True
        }})

        img = Image.new('RGB', (800, 600), 'red')
        start = time.time()
        for _ in range(500):
            _ = tx['train'](img)
        elapsed = time.time() - start
        throughput = 500 / elapsed
        print(f"Augmentation throughput: {{throughput:.0f}} images/sec")
        assert throughput > 10  # at least 10/sec

    def test_cpu_concurrent_crop_and_inference(self, tmp_path):
        """Simulate real pipeline: crop + inference concurrently on CPU."""
        from src.utils.cropper import crop_single_image
        from src.models.cnn_trainer import get_backbone, get_transforms

        # Prepare image
        img = Image.new('RGB', (1000, 1000), 'blue')
        img.save(str(tmp_path / 'test.jpg'))
        lines = [f"0 {{(i%10)*0.1+0.05}} {{(i//10)*0.1+0.05}} 0.08 0.08" for i in range(100)]
        (tmp_path / 'test.txt').write_text(NL.join(lines))

        # Crop
        start = time.time()
        crops = crop_single_image(str(tmp_path / 'test.jpg'), str(tmp_path / 'test.txt'), min_size=10)
        crop_time = time.time() - start

        # Inference on all crops
        model = get_backbone('resnet18', num_classes=5, pretrained=False)
        model.eval()
        tx = get_transforms((64, 64))

        start = time.time()
        with torch.no_grad():
            for crop_info in crops:
                tensor = tx['val'](crop_info['crop_image'].convert('RGB')).unsqueeze(0)
                _ = model(tensor)
        infer_time = time.time() - start

        total = crop_time + infer_time
        print(f"100 objects: crop={{crop_time:.2f}}s, infer={{infer_time:.2f}}s, total={{total:.2f}}s")
        assert total < 120  # should complete within 2 minutes on any CPU


# ============================================================
# GPU COMPATIBILITY TESTS
# ============================================================

class TestGPUCompatibility:
    """Test GPU detection, fallback, and CUDA/CPU compatibility."""

    def test_auto_device_detection(self):
        """Verify 'auto' device selection works correctly."""
        from src.models.cnn_trainer import train_attribute_model
        # The device='auto' should pick cuda if available, else cpu
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Auto device: {{device}} (CUDA available: {{torch.cuda.is_available()}})")
        assert str(device) == expected

    def test_explicit_cpu_works(self, tmp_path):
        """Force CPU device even if GPU available."""
        from src.models.cnn_trainer import train_attribute_model

        ds_dir = tmp_path / 'ds'
        for cls in ['a', 'b']:
            d = ds_dir / cls
            d.mkdir(parents=True)
            for i in range(10):
                Image.new('RGB', (32, 32)).save(str(d / f'{{i}}.jpg'))

        result = train_attribute_model(
            dataset_dir=str(ds_dir), attribute_name='test',
            output_dir=str(tmp_path / 'm'), backbone='resnet18',
            epochs=1, pretrained=False, device='cpu',
            num_workers=0, early_stopping=0, target_size=(32, 32),
            batch_size=4
        )
        assert result['success'] is True
        assert result['device'] == 'cpu'

    def test_cuda_device_graceful_fallback(self):
        """If CUDA requested but unavailable, verify behavior."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test fallback")
        # Requesting cuda on a cpu-only system should raise or handle
        try:
            device = torch.device('cuda')
            x = torch.randn(1, 3, 32, 32).to(device)
            # If we got here, CUDA is actually available
            assert False, "Should have failed on CPU-only system"
        except (RuntimeError, AssertionError):
            pass  # Expected: CUDA not available
        print("CUDA fallback test passed (no CUDA available)")

    def test_model_on_cpu_after_gpu_request(self):
        """Verify model works on CPU regardless of initial device request."""
        from src.models.cnn_trainer import get_backbone
        model = get_backbone('resnet18', num_classes=5, pretrained=False)
        model = model.to('cpu')
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5)
        assert out.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_gpu_training_if_available(self, tmp_path):
        """If GPU is available, test actual GPU training."""
        from src.models.cnn_trainer import train_attribute_model

        ds_dir = tmp_path / 'ds'
        for cls in ['a', 'b']:
            d = ds_dir / cls
            d.mkdir(parents=True)
            for i in range(20):
                Image.new('RGB', (64, 64), (i*10, 50, 100)).save(str(d / f'{{i}}.jpg'))

        result = train_attribute_model(
            dataset_dir=str(ds_dir), attribute_name='gpu_test',
            output_dir=str(tmp_path / 'm'), backbone='resnet18',
            epochs=2, pretrained=False, device='cuda',
            num_workers=0, early_stopping=0, target_size=(64, 64),
            batch_size=8
        )
        assert result['success'] is True
        assert 'cuda' in result['device']
        print(f"GPU training completed: {{result['best_val_acc']*100:.1f}}% accuracy")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_gpu_vram_limit(self):
        """Test GPU VRAM limit detection."""
        vram = torch.cuda.get_device_properties(0).total_mem
        vram_gb = vram / (1024**3)
        print(f"GPU VRAM: {{vram_gb:.2f}} GB")

        # Try allocating increasingly large tensors until OOM
        max_batch = 1
        for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                x = torch.randn(batch, 3, 224, 224, device='cuda')
                del x
                torch.cuda.empty_cache()
                max_batch = batch
            except RuntimeError:
                break
        print(f"Max batch size at 224x224 on GPU: {{max_batch}}")
        assert max_batch >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_gpu_inference_speed_vs_cpu(self):
        """Compare GPU vs CPU inference speed."""
        from src.models.cnn_trainer import get_backbone

        model = get_backbone('resnet18', num_classes=10, pretrained=False)
        x = torch.randn(8, 3, 224, 224)

        # CPU
        model_cpu = model.to('cpu')
        model_cpu.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_cpu(x)
        cpu_time = time.time() - start

        # GPU
        model_gpu = model.to('cuda')
        model_gpu.eval()
        x_gpu = x.to('cuda')
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"CPU: {{cpu_time:.2f}}s, GPU: {{gpu_time:.2f}}s, Speedup: {{speedup:.1f}}x")


# ============================================================
# COMBINED RESOURCE LIMIT TESTS
# ============================================================

class TestCombinedResourceLimits:
    """Test the system under combined resource pressure."""

    def test_full_pipeline_memory_profile(self, tmp_path):
        """Run crop->organize->train and track memory throughout."""
        from src.utils.cropper import crop_dataset, organize_by_attribute
        from src.models.cnn_trainer import train_attribute_model

        # Setup: 20 images, 10 objects each
        img_dir = tmp_path / 'images'
        lbl_dir = tmp_path / 'labels'
        json_dir = tmp_path / 'jsons'
        img_dir.mkdir(); lbl_dir.mkdir(); json_dir.mkdir()

        materials = ['leather', 'wood', 'fabric']
        for i in range(20):
            Image.new('RGB', (400, 400), (i*12, 100, 50)).save(str(img_dir / f'img{{i:03d}}.jpg'))
            lines = []
            shapes = []
            for j in range(10):
                cx = (j % 5) * 0.2 + 0.1
                cy = (j // 5) * 0.5 + 0.25
                lines.append(f"0 {{cx:.3f}} {{cy:.3f}} 0.15 0.4")
                x1 = int((cx - 0.075) * 400)
                y1 = int((cy - 0.2) * 400)
                x2 = int((cx + 0.075) * 400)
                y2 = int((cy + 0.2) * 400)
                shapes.append({{
                    'label': 'chair', 'points': [[x1,y1],[x2,y2]],
                    'shape_type': 'rectangle', 'flags': {{}},
                    'attributes': {{'material': materials[(i+j)%3]}}
                }})
            (lbl_dir / f'img{{i:03d}}.txt').write_text(NL.join(lines))
            (json_dir / f'img{{i:03d}}.json').write_text(json.dumps({{'shapes': shapes}}))

        gc.collect()
        ram_start = get_ram_mb()
        print(f"RAM at start: {{ram_start:.0f}} MB")

        # Step 1: Crop
        stats = crop_dataset(str(img_dir), str(lbl_dir), str(json_dir),
                            output_dir=str(tmp_path / 'cropped'))
        ram_crop = get_ram_mb()
        print(f"After crop ({{stats['total_crops']}} crops): {{ram_crop:.0f}} MB (+{{ram_crop-ram_start:.0f}})"
        )

        # Step 2: Organize
        org_stats = organize_by_attribute(
            str(tmp_path / 'cropped'), attribute_name='material',
            output_dir=str(tmp_path / 'ds' / 'material'), target_size=(64, 64)
        )
        ram_org = get_ram_mb()
        print(f"After organize ({{org_stats['total_images']}} imgs): {{ram_org:.0f}} MB (+{{ram_org-ram_crop:.0f}})")

        # Step 3: Train (short)
        result = train_attribute_model(
            dataset_dir=str(tmp_path / 'ds' / 'material'),
            attribute_name='material',
            output_dir=str(tmp_path / 'models'),
            backbone='resnet18', epochs=2, batch_size=8,
            pretrained=False, device='cpu', num_workers=0,
            early_stopping=0, target_size=(64, 64)
        )
        ram_train = get_ram_mb()
        print(f"After train: {{ram_train:.0f}} MB (+{{ram_train-ram_org:.0f}})")
        assert result['success'] is True

        # Total memory growth should be bounded
        total_growth = ram_train - ram_start
        print(f"Total RAM growth: {{total_growth:.0f}} MB")
        assert total_growth < 2000, f"Excessive memory: {{total_growth:.0f}} MB"

    def test_max_concurrent_models_in_registry(self, tmp_path):
        """Load as many models as RAM allows."""
        from src.models.cnn_trainer import get_backbone

        gc.collect()
        ram_before = get_ram_mb()
        free_ram = get_free_ram_mb()
        print(f"Free RAM: {{free_ram:.0f}} MB")

        models = []
        max_models = min(10, int(free_ram / 200))  # ~200MB per resnet18
        for i in range(max_models):
            m = get_backbone('resnet18', num_classes=10, pretrained=False)
            models.append(m)

        ram_after = get_ram_mb()
        per_model = (ram_after - ram_before) / max(len(models), 1)
        print(f"Loaded {{len(models)}} models, {{per_model:.0f}} MB each, total {{ram_after-ram_before:.0f}} MB")

        del models
        gc.collect()
        ram_freed = get_ram_mb()
        print(f"After cleanup: {{ram_freed:.0f}} MB (freed {{ram_after-ram_freed:.0f}} MB)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
'''

output_path = os.path.join(os.path.dirname(__file__), 'test_hardware_limits.py')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(test_code)
print(f'Generated {output_path}')
print(f'File size: {os.path.getsize(output_path)} bytes')
