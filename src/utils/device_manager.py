"""Device manager: CPU/GPU selection, resource allocation, and diagnostics.

Provides centralized control over compute device selection,
memory management, and hardware capability detection.
"""
import os
import sys
import gc
import platform
from typing import Any, Dict, List, Optional, Tuple

import torch


class DeviceManager:
    """
    Manages CPU/GPU device selection and resource allocation.
    
    Features:
    - Auto-detect best available device
    - Force CPU or GPU mode
    - GPU VRAM monitoring
    - CPU/RAM usage tracking
    - Optimal batch size estimation
    - Mixed precision support detection
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Args:
            device: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        """
        self._requested = device
        self._device = self._resolve_device(device)
        self._gpu_info = self._detect_gpu() if self._device.type == 'cuda' else None
    
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(f"[DeviceManager] CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
            return torch.device(device)
        else:
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
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system hardware info."""
        info = {
            'platform': platform.platform(),
            'python': sys.version,
            'torch': torch.__version__,
            'device_requested': self._requested,
            'device_active': str(self._device),
            'cpu': {
                'count_physical': os.cpu_count(),
                'torch_threads': torch.get_num_threads(),
                'torch_interop_threads': torch.get_num_interop_threads(),
            },
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # RAM info
        try:
            import psutil
            vm = psutil.virtual_memory()
            info['ram'] = {
                'total_mb': vm.total // (1024 * 1024),
                'available_mb': vm.available // (1024 * 1024),
                'used_percent': vm.percent,
            }
        except ImportError:
            info['ram'] = {'note': 'psutil not installed'}
        
        # GPU info
        if self._gpu_info:
            info['gpu'] = self._gpu_info
            info['gpu']['free_memory_mb'] = self.get_gpu_free_memory_mb()
            info['gpu']['supports_fp16'] = self.supports_fp16()
            info['gpu']['supports_bf16'] = self.supports_bf16()
        
        return info
    
    def get_gpu_free_memory_mb(self) -> int:
        """Get free GPU memory in MB."""
        if not self.is_gpu:
            return 0
        idx = self._device.index or 0
        free, total = torch.cuda.mem_get_info(idx)
        return free // (1024 * 1024)
    
    def get_gpu_used_memory_mb(self) -> int:
        """Get used GPU memory in MB."""
        if not self.is_gpu:
            return 0
        return torch.cuda.memory_allocated(self._device) // (1024 * 1024)
    
    def supports_fp16(self) -> bool:
        """Check if GPU supports FP16 (half precision)."""
        if not self.is_gpu or not self._gpu_info:
            return False
        return self._gpu_info['major'] >= 7  # Volta+
    
    def supports_bf16(self) -> bool:
        """Check if GPU supports BF16."""
        if not self.is_gpu or not self._gpu_info:
            return False
        return self._gpu_info['major'] >= 8  # Ampere+
    
    def estimate_max_batch_size(
        self,
        model: torch.nn.Module,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        safety_factor: float = 0.8
    ) -> int:
        """
        Estimate the maximum batch size that fits in available memory.
        
        Args:
            model: The PyTorch model
            input_size: (C, H, W) input tensor shape
            safety_factor: Fraction of free memory to use (0-1)
        
        Returns:
            Estimated max batch size
        """
        if self.is_cpu:
            # For CPU, estimate based on available RAM
            try:
                import psutil
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
            except ImportError:
                available_mb = 4096  # conservative default
            
            # Estimate model + data memory per sample
            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            sample_mb = (input_size[0] * input_size[1] * input_size[2] * 4) / (1024 * 1024)  # float32
            
            usable_mb = available_mb * safety_factor - param_mb * 2  # model + gradients
            max_batch = max(1, int(usable_mb / (sample_mb * 3)))  # input + activations + gradients
            return min(max_batch, 256)  # cap at 256
        
        else:
            # For GPU, use VRAM
            free_mb = self.get_gpu_free_memory_mb()
            if free_mb == 0:
                return 8  # fallback
            
            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            sample_mb = (input_size[0] * input_size[1] * input_size[2] * 4) / (1024 * 1024)
            
            usable_mb = free_mb * safety_factor - param_mb * 2
            max_batch = max(1, int(usable_mb / (sample_mb * 4)))  # more memory per sample on GPU due to activations
            return min(max_batch, 512)
    
    def optimize_for_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply device-specific optimizations to a model.
        
        CPU: Set optimal thread count
        GPU: Move to GPU, enable cudnn benchmark, optional FP16
        """
        model = model.to(self._device)
        
        if self.is_cpu:
            # Use all available CPU cores for PyTorch
            cpu_count = os.cpu_count() or 1
            try:
                torch.set_num_threads(cpu_count)
            except RuntimeError:
                pass  # already set
            try:
                torch.set_num_interop_threads(max(1, cpu_count // 2))
            except RuntimeError:
                pass  # can only be set once before parallel work starts
        
        elif self.is_gpu:
            # Enable cudnn benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable TF32 on Ampere+ for faster computation
            if self._gpu_info and self._gpu_info['major'] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        return model
    
    def get_optimal_num_workers(self) -> int:
        """Get optimal DataLoader num_workers for this device."""
        cpu_count = os.cpu_count() or 1
        if self.is_gpu:
            # GPU: use more workers to keep GPU fed
            return min(8, cpu_count)
        else:
            # CPU: fewer workers to leave cores for training
            return min(4, max(0, cpu_count - 2))
    
    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get optimal DataLoader keyword arguments for this device."""
        return {
            'num_workers': self.get_optimal_num_workers(),
            'pin_memory': self.is_gpu,
            'persistent_workers': self.get_optimal_num_workers() > 0,
        }
    
    def clear_memory(self):
        """Free unused memory on the active device."""
        gc.collect()
        if self.is_gpu:
            torch.cuda.empty_cache()
    
    def memory_summary(self) -> str:
        """Get a human-readable memory summary."""
        lines = []
        if self.is_gpu:
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
        
        try:
            import psutil
            vm = psutil.virtual_memory()
            lines.append(f"RAM: {vm.total//(1024**2)} MB total, {vm.available//(1024**2)} MB free ({vm.percent}% used)")
            proc = psutil.Process(os.getpid())
            lines.append(f"Process RSS: {proc.memory_info().rss//(1024**2)} MB")
        except ImportError:
            pass
        
        lines.append(f"Device: {self._device}")
        lines.append(f"Torch threads: {torch.get_num_threads()}")
        
        return chr(10).join(lines)


# Singleton
_device_manager: Optional[DeviceManager] = None

def get_device_manager(device: str = 'auto') -> DeviceManager:
    """Get or create the singleton DeviceManager."""
    global _device_manager
    if _device_manager is None or device != 'auto':
        _device_manager = DeviceManager(device)
    return _device_manager
