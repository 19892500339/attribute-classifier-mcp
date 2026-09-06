"""Device manager: CPU/GPU selection, resource allocation, adaptive training.

Provides centralized control over compute device selection,
memory management, hardware capability detection, and
adaptive training intensity that caps at a configurable
percentage of available resources (default 60%).
"""
import os
import sys
import gc
import platform
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DeviceManager:
    """
    Manages CPU/GPU device selection, resource allocation, and adaptive training.
    
    Modes:
    - 'auto': detect best device, apply adaptive resource limits
    - 'cpu': force CPU, maximize RAM + threads for training
    - 'cuda'/'cuda:0'/etc: force GPU, maximize VRAM + CUDA for training
    
    Adaptive training:
    - Profiles hardware (CPU cores, RAM, GPU VRAM)
    - Caps resource usage at a configurable percentage (default 60%)
    - Auto-tunes: batch_size, num_workers, image_size, accumulation_steps
    """
    
    def __init__(self, device: str = 'auto', max_resource_percent: float = 60.0):
        """
        Args:
            device: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'
            max_resource_percent: Max percentage of resources to use (0-100, default 60)
        """
        self._requested = device
        self._max_percent = max(1.0, min(100.0, max_resource_percent))
        self._device = self._resolve_device(device)
        self._gpu_info = self._detect_gpu() if self._device.type == 'cuda' else None
        self._hw_profile = self._profile_hardware()
    
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print("[DeviceManager] CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
            return torch.device(device)
        return torch.device('cpu')
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def is_gpu(self) -> bool:
        return self._device.type == 'cuda'
    
    @property
    def is_cpu(self) -> bool:
        return self._device.type == 'cpu'
    
    @property
    def max_resource_percent(self) -> float:
        return self._max_percent
    
    def _detect_gpu(self) -> Optional[Dict[str, Any]]:
        """Detect GPU capabilities."""
        if not torch.cuda.is_available():
            return None
        idx = self._device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return {
            'name': props.name,
            'total_memory_mb': props.total_mem // (1024 * 1024),
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        }
    
    def _profile_hardware(self) -> Dict[str, Any]:
        """Profile all hardware resources."""
        profile = {
            'cpu_cores_physical': os.cpu_count() or 1,
            'cpu_cores_logical': os.cpu_count() or 1,
        }
        
        # RAM
        try:
            import psutil
            vm = psutil.virtual_memory()
            profile['ram_total_mb'] = vm.total // (1024 * 1024)
            profile['ram_available_mb'] = vm.available // (1024 * 1024)
            profile['ram_used_percent'] = vm.percent
            cpu_info = psutil.cpu_freq()
            if cpu_info:
                profile['cpu_freq_mhz'] = cpu_info.max or cpu_info.current
        except ImportError:
            profile['ram_total_mb'] = 8192  # conservative fallback
            profile['ram_available_mb'] = 4096
            profile['ram_used_percent'] = 50.0
        
        # GPU
        if self.is_gpu and self._gpu_info:
            idx = self._device.index or 0
            free, total = torch.cuda.mem_get_info(idx)
            profile['gpu_total_mb'] = total // (1024 * 1024)
            profile['gpu_free_mb'] = free // (1024 * 1024)
            profile['gpu_name'] = self._gpu_info['name']
            profile['gpu_sm_count'] = self._gpu_info['multi_processor_count']
        
        return profile
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system hardware info."""
        info = {
            'platform': platform.platform(),
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'device_requested': self._requested,
            'device_active': str(self._device),
            'max_resource_percent': self._max_percent,
            'cpu': {
                'count': self._hw_profile.get('cpu_cores_physical', 0),
                'freq_mhz': self._hw_profile.get('cpu_freq_mhz', 0),
                'torch_threads': torch.get_num_threads(),
            },
            'ram': {
                'total_mb': self._hw_profile.get('ram_total_mb', 0),
                'available_mb': self._hw_profile.get('ram_available_mb', 0),
                'used_percent': self._hw_profile.get('ram_used_percent', 0),
            },
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if self._gpu_info:
            info['gpu'] = {
                **self._gpu_info,
                'free_memory_mb': self._hw_profile.get('gpu_free_mb', 0),
                'supports_fp16': self.supports_fp16(),
                'supports_bf16': self.supports_bf16(),
            }
        return info
    
    def get_gpu_free_memory_mb(self) -> int:
        if not self.is_gpu:
            return 0
        idx = self._device.index or 0
        free, _ = torch.cuda.mem_get_info(idx)
        return free // (1024 * 1024)
    
    def get_gpu_used_memory_mb(self) -> int:
        if not self.is_gpu:
            return 0
        return torch.cuda.memory_allocated(self._device) // (1024 * 1024)
    
    def supports_fp16(self) -> bool:
        if not self.is_gpu or not self._gpu_info:
            return False
        return self._gpu_info['major'] >= 7
    
    def supports_bf16(self) -> bool:
        if not self.is_gpu or not self._gpu_info:
            return False
        return self._gpu_info['major'] >= 8
    
    # ================================================================
    # ADAPTIVE TRAINING CONFIGURATION
    # ================================================================
    
    def adaptive_training_config(
        self,
        backbone: str = 'resnet18',
        num_classes: int = 10,
        dataset_size: int = 1000,
        base_image_size: int = 224,
        max_resource_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Auto-tune all training hyperparameters based on hardware profiling.
        Caps resource usage at max_resource_percent of available resources.
        
        Returns a dict with recommended:
        - batch_size, image_size, num_workers, learning_rate
        - accumulation_steps (for effective larger batches)
        - epochs (scaled by dataset size)
        - pin_memory, mixed_precision
        - resource_report with utilization estimates
        """
        pct = (max_resource_percent or self._max_percent) / 100.0
        
        # Get model memory footprint
        model_memory_mb = self._estimate_model_memory_mb(backbone, num_classes)
        
        if self.is_gpu:
            return self._adaptive_gpu_config(
                backbone, num_classes, dataset_size, base_image_size,
                model_memory_mb, pct
            )
        else:
            return self._adaptive_cpu_config(
                backbone, num_classes, dataset_size, base_image_size,
                model_memory_mb, pct
            )
    
    def _estimate_model_memory_mb(self, backbone: str, num_classes: int) -> float:
        """Estimate model memory in MB (params + gradients + optimizer)."""
        # Approximate parameter counts per backbone
        param_counts = {
            'resnet18': 11.7e6,
            'resnet34': 21.8e6,
            'resnet50': 25.6e6,
            'mobilenet_v2': 3.5e6,
            'efficientnet_b0': 5.3e6,
        }
        params = param_counts.get(backbone, 11.7e6)
        # Memory: params(fp32) + gradients(fp32) + optimizer_state(Adam=2x)
        bytes_per_param = 4  # float32
        total_bytes = params * bytes_per_param * 4  # param + grad + 2x adam
        return total_bytes / (1024 * 1024)
    
    def _estimate_activation_memory_mb(self, backbone: str, batch_size: int, img_size: int) -> float:
        """Estimate activation memory during forward/backward pass."""
        # Rough multiplier: activations scale with batch * img_size^2
        activation_multipliers = {
            'resnet18': 0.8,
            'resnet34': 1.2,
            'resnet50': 2.5,
            'mobilenet_v2': 0.6,
            'efficientnet_b0': 0.7,
        }
        mult = activation_multipliers.get(backbone, 1.0)
        # MB = batch * (img/224)^2 * multiplier * base_mb
        base_mb = 50  # rough base for batch=1, 224x224
        return batch_size * (img_size / 224.0) ** 2 * mult * base_mb / batch_size * batch_size
    
    def _adaptive_gpu_config(
        self, backbone, num_classes, dataset_size, base_image_size,
        model_memory_mb, pct
    ) -> Dict[str, Any]:
        """Compute adaptive config for GPU training."""
        gpu_total = self._hw_profile.get('gpu_total_mb', 4096)
        gpu_free = self._hw_profile.get('gpu_free_mb', gpu_total)
        
        # Usable VRAM = free * percent_limit
        usable_vram = gpu_free * pct
        
        # Reserve for model + optimizer
        vram_for_data = usable_vram - model_memory_mb
        if vram_for_data < 100:
            vram_for_data = 100
        
        # Estimate per-sample activation memory
        sample_mb = self._estimate_activation_memory_mb(backbone, 1, base_image_size)
        
        # Max batch size from VRAM
        max_batch = max(1, int(vram_for_data / sample_mb))
        
        # Pick a power-of-2-friendly batch size
        batch_size = min(max_batch, 128)
        for target in [128, 64, 32, 16, 8, 4, 2, 1]:
            if target <= max_batch:
                batch_size = target
                break
        
        # Accumulation steps for effective batch if too small
        effective_batch = 32
        accumulation_steps = max(1, effective_batch // batch_size)
        
        # Image size: reduce if VRAM too tight
        image_size = base_image_size
        if max_batch < 4:
            image_size = max(64, base_image_size // 2)
            # Recalculate with smaller image
            sample_mb = self._estimate_activation_memory_mb(backbone, 1, image_size)
            max_batch = max(1, int(vram_for_data / sample_mb))
            for target in [64, 32, 16, 8, 4, 2, 1]:
                if target <= max_batch:
                    batch_size = target
                    break
        
        # Workers: use CPU cores to feed GPU
        cpu_count = self._hw_profile.get('cpu_cores_physical', 4)
        num_workers = min(8, cpu_count)
        
        # Learning rate scales with effective batch size
        base_lr = 0.001
        lr = base_lr * (batch_size * accumulation_steps) / 32.0
        lr = max(0.0001, min(0.01, lr))
        
        # Epochs: more data = fewer epochs needed
        if dataset_size < 100:
            epochs = 100
        elif dataset_size < 500:
            epochs = 50
        elif dataset_size < 2000:
            epochs = 30
        else:
            epochs = 20
        
        # Mixed precision
        use_fp16 = self.supports_fp16()
        
        # Estimated memory usage
        estimated_vram_usage = model_memory_mb + sample_mb * batch_size
        estimated_vram_pct = (estimated_vram_usage / gpu_total) * 100 if gpu_total > 0 else 0
        
        return {
            'device': str(self._device),
            'mode': 'gpu',
            'batch_size': batch_size,
            'image_size': [image_size, image_size],
            'num_workers': num_workers,
            'learning_rate': round(lr, 6),
            'epochs': epochs,
            'accumulation_steps': accumulation_steps,
            'effective_batch_size': batch_size * accumulation_steps,
            'pin_memory': True,
            'mixed_precision': use_fp16,
            'backbone': backbone,
            'resource_limit_percent': self._max_percent,
            'resource_report': {
                'gpu_name': self._hw_profile.get('gpu_name', 'unknown'),
                'gpu_total_mb': gpu_total,
                'gpu_free_mb': gpu_free,
                'gpu_usable_mb': round(usable_vram),
                'model_memory_mb': round(model_memory_mb),
                'estimated_usage_mb': round(estimated_vram_usage),
                'estimated_usage_percent': round(estimated_vram_pct, 1),
                'headroom_mb': round(usable_vram - estimated_vram_usage),
            }
        }
    
    def _adaptive_cpu_config(
        self, backbone, num_classes, dataset_size, base_image_size,
        model_memory_mb, pct
    ) -> Dict[str, Any]:
        """Compute adaptive config for CPU training."""
        ram_total = self._hw_profile.get('ram_total_mb', 8192)
        ram_available = self._hw_profile.get('ram_available_mb', 4096)
        cpu_count = self._hw_profile.get('cpu_cores_physical', 4)
        
        # Usable RAM = available * percent_limit
        usable_ram = ram_available * pct
        
        # Reserve for model + optimizer
        ram_for_data = usable_ram - model_memory_mb
        if ram_for_data < 200:
            ram_for_data = 200
        
        # Per-sample memory on CPU (less than GPU due to no activation caching)
        sample_bytes = 3 * base_image_size * base_image_size * 4  # input tensor
        sample_mb = sample_bytes / (1024 * 1024) * 3  # input + grad + overhead
        
        # Max batch from RAM
        max_batch = max(1, int(ram_for_data / sample_mb))
        
        # Pick reasonable batch size (CPU training is slower, smaller batches ok)
        batch_size = min(max_batch, 64)
        for target in [64, 32, 16, 8, 4, 2, 1]:
            if target <= max_batch:
                batch_size = target
                break
        
        # Image size: reduce if RAM tight
        image_size = base_image_size
        if max_batch < 4:
            image_size = max(64, base_image_size // 2)
            sample_bytes = 3 * image_size * image_size * 4
            sample_mb = sample_bytes / (1024 * 1024) * 3
            max_batch = max(1, int(ram_for_data / sample_mb))
            for target in [32, 16, 8, 4, 2, 1]:
                if target <= max_batch:
                    batch_size = target
                    break
        
        accumulation_steps = max(1, 32 // batch_size)
        
        # Workers: leave some cores free for training
        usable_cores = max(1, int(cpu_count * pct))
        train_threads = max(1, usable_cores - 2)
        num_workers = min(4, max(0, usable_cores - train_threads))
        
        # Learning rate
        base_lr = 0.001
        lr = base_lr * (batch_size * accumulation_steps) / 32.0
        lr = max(0.0001, min(0.01, lr))
        
        # Epochs
        if dataset_size < 100:
            epochs = 100
        elif dataset_size < 500:
            epochs = 50
        elif dataset_size < 2000:
            epochs = 30
        else:
            epochs = 20
        
        # Estimated memory usage
        estimated_ram_usage = model_memory_mb + sample_mb * batch_size
        estimated_ram_pct = (estimated_ram_usage / ram_total) * 100 if ram_total > 0 else 0
        
        return {
            'device': 'cpu',
            'mode': 'cpu',
            'batch_size': batch_size,
            'image_size': [image_size, image_size],
            'num_workers': num_workers,
            'learning_rate': round(lr, 6),
            'epochs': epochs,
            'accumulation_steps': accumulation_steps,
            'effective_batch_size': batch_size * accumulation_steps,
            'pin_memory': False,
            'mixed_precision': False,
            'backbone': backbone,
            'torch_threads': train_threads,
            'resource_limit_percent': self._max_percent,
            'resource_report': {
                'cpu_cores': cpu_count,
                'cpu_cores_for_training': train_threads,
                'ram_total_mb': ram_total,
                'ram_available_mb': ram_available,
                'ram_usable_mb': round(usable_ram),
                'model_memory_mb': round(model_memory_mb),
                'estimated_usage_mb': round(estimated_ram_usage),
                'estimated_usage_percent': round(estimated_ram_pct, 1),
                'headroom_mb': round(usable_ram - estimated_ram_usage),
            }
        }
    
    # ================================================================
    # DEDICATED MODE METHODS
    # ================================================================
    
    def apply_cpu_dedicated_mode(self) -> Dict[str, Any]:
        """
        Force all compute and memory to CPU.
        Maximizes CPU threads, frees GPU memory if any.
        """
        cpu_count = self._hw_profile.get('cpu_cores_physical', 4)
        usable_cores = max(1, int(cpu_count * self._max_percent / 100.0))
        
        try:
            torch.set_num_threads(usable_cores)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(max(1, usable_cores // 2))
        except RuntimeError:
            pass
        
        # Free any GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'mode': 'cpu_dedicated',
            'device': 'cpu',
            'torch_threads': usable_cores,
            'resource_limit_percent': self._max_percent,
            'ram_available_mb': self._hw_profile.get('ram_available_mb', 0),
        }
    
    def apply_gpu_dedicated_mode(self) -> Dict[str, Any]:
        """
        Force all compute and memory to GPU.
        Enables cudnn benchmark, TF32, and clears GPU cache.
        """
        if not self.is_gpu:
            return {
                'mode': 'gpu_dedicated',
                'error': 'No CUDA GPU available. Use cpu mode.',
                'device': 'cpu'
            }
        
        # Enable maximum GPU performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        if self._gpu_info and self._gpu_info['major'] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Clear stale allocations
        torch.cuda.empty_cache()
        gc.collect()
        
        idx = self._device.index or 0
        free, total = torch.cuda.mem_get_info(idx)
        
        return {
            'mode': 'gpu_dedicated',
            'device': str(self._device),
            'gpu_name': self._gpu_info['name'],
            'gpu_total_mb': total // (1024 * 1024),
            'gpu_free_mb': free // (1024 * 1024),
            'gpu_usable_mb': int((free // (1024 * 1024)) * self._max_percent / 100.0),
            'cudnn_benchmark': True,
            'tf32_enabled': self._gpu_info['major'] >= 8 if self._gpu_info else False,
            'fp16_available': self.supports_fp16(),
            'resource_limit_percent': self._max_percent,
        }
    
    # ================================================================
    # EXISTING METHODS (kept compatible)
    # ================================================================
    
    def estimate_max_batch_size(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        safety_factor: float = 0.8
    ) -> int:
        """Estimate max batch size that fits in available memory."""
        pct = self._max_percent / 100.0
        
        if self.is_cpu:
            try:
                import psutil
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
            except ImportError:
                available_mb = 4096
            
            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            sample_mb = (input_size[0] * input_size[1] * input_size[2] * 4) / (1024 * 1024)
            usable_mb = available_mb * pct * safety_factor - param_mb * 2
            max_batch = max(1, int(usable_mb / (sample_mb * 3)))
            return min(max_batch, 256)
        else:
            free_mb = self.get_gpu_free_memory_mb()
            if free_mb == 0:
                return 8
            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            sample_mb = (input_size[0] * input_size[1] * input_size[2] * 4) / (1024 * 1024)
            usable_mb = free_mb * pct * safety_factor - param_mb * 2
            max_batch = max(1, int(usable_mb / (sample_mb * 4)))
            return min(max_batch, 512)
    
    def optimize_for_device(self, model: nn.Module) -> nn.Module:
        """Apply device-specific optimizations to a model."""
        model = model.to(self._device)
        
        if self.is_cpu:
            cpu_count = os.cpu_count() or 1
            usable = max(1, int(cpu_count * self._max_percent / 100.0))
            try:
                torch.set_num_threads(usable)
            except RuntimeError:
                pass
            try:
                torch.set_num_interop_threads(max(1, usable // 2))
            except RuntimeError:
                pass
        elif self.is_gpu:
            torch.backends.cudnn.benchmark = True
            if self._gpu_info and self._gpu_info['major'] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        return model
    
    def get_optimal_num_workers(self) -> int:
        cpu_count = os.cpu_count() or 1
        usable = max(1, int(cpu_count * self._max_percent / 100.0))
        if self.is_gpu:
            return min(8, usable)
        return min(4, max(0, usable - 2))
    
    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        nw = self.get_optimal_num_workers()
        return {
            'num_workers': nw,
            'pin_memory': self.is_gpu,
            'persistent_workers': nw > 0,
        }
    
    def clear_memory(self):
        gc.collect()
        if self.is_gpu:
            torch.cuda.empty_cache()
    
    def memory_summary(self) -> str:
        lines = []
        if self.is_gpu and self._gpu_info:
            idx = self._device.index or 0
            total = torch.cuda.get_device_properties(idx).total_mem / (1024**2)
            alloc = torch.cuda.memory_allocated(idx) / (1024**2)
            cached = torch.cuda.memory_reserved(idx) / (1024**2)
            free = self.get_gpu_free_memory_mb()
            lines.append(f"GPU: {self._gpu_info['name']}")
            lines.append(f"  Total VRAM:     {total:.0f} MB")
            lines.append(f"  Allocated:      {alloc:.0f} MB")
            lines.append(f"  Cached:         {cached:.0f} MB")
            lines.append(f"  Free:           {free:.0f} MB")
            lines.append(f"  Usable ({self._max_percent:.0f}%):  {free * self._max_percent / 100:.0f} MB")
        
        try:
            import psutil
            vm = psutil.virtual_memory()
            lines.append(f"RAM: {vm.total//(1024**2)} MB total, {vm.available//(1024**2)} MB free ({vm.percent}% used)")
            lines.append(f"RAM usable ({self._max_percent:.0f}%): {int(vm.available//(1024**2) * self._max_percent / 100)} MB")
            proc = psutil.Process(os.getpid())
            lines.append(f"Process RSS: {proc.memory_info().rss//(1024**2)} MB")
        except ImportError:
            lines.append("RAM: psutil not installed")
        
        lines.append(f"Device: {self._device} | Limit: {self._max_percent:.0f}%")
        lines.append(f"Torch threads: {torch.get_num_threads()}")
        
        return chr(10).join(lines)


# Singleton
_device_manager: Optional[DeviceManager] = None

def get_device_manager(device: str = 'auto', max_resource_percent: float = 60.0) -> DeviceManager:
    """Get or create the singleton DeviceManager."""
    global _device_manager
    if _device_manager is None or device != 'auto':
        _device_manager = DeviceManager(device, max_resource_percent)
    return _device_manager
