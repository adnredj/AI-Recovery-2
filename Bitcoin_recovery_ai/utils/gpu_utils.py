from typing import Optional, List, Dict, Union
import logging
import torch
import torch.cuda as cuda
from ..utils.logging import setup_logger

def setup_gpu(config: Dict[str, any]) -> torch.device:
    """Setup GPU device based on configuration
    
    Args:
        config: Configuration dictionary containing GPU settings
        
    Returns:
        torch.device: Selected device (CPU or GPU)
    """
    logger = setup_logger(__name__)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device('cpu')
        
    try:
        # Get GPU settings from config
        gpu_id = config.get('gpu_id', 0)
        memory_fraction = config.get('gpu_memory_fraction', 0.8)
        
        # Validate GPU ID
        if gpu_id >= torch.cuda.device_count():
            logger.warning(f"GPU {gpu_id} not found, using GPU 0")
            gpu_id = 0
            
        # Set device
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        # Set memory fraction
        if memory_fraction > 0 and memory_fraction <= 1:
            _set_gpu_memory_fraction(memory_fraction, gpu_id)
            
        # Log GPU info
        _log_gpu_info(gpu_id)
        
        return device
        
    except Exception as e:
        logger.error(f"Error setting up GPU: {str(e)}")
        logger.warning("Falling back to CPU")
        return torch.device('cpu')

def get_available_gpus() -> List[Dict[str, any]]:
    """Get information about available GPUs
    
    Returns:
        List of dictionaries containing GPU information
    """
    gpus = []
    
    if not torch.cuda.is_available():
        return gpus
        
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'id': i,
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            })
    except Exception:
        pass
        
    return gpus

def _set_gpu_memory_fraction(fraction: float, device_id: int = 0):
    """Set maximum GPU memory fraction to use
    
    Args:
        fraction: Fraction of total memory to use (0-1)
        device_id: GPU device ID
    """
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            max_memory = int(total_memory * fraction)
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            torch.cuda.set_per_process_memory_limit(max_memory)
    except Exception:
        pass

def _log_gpu_info(device_id: int):
    """Log information about selected GPU
    
    Args:
        device_id: GPU device ID
    """
    logger = setup_logger(__name__)
    
    try:
        props = torch.cuda.get_device_properties(device_id)
        logger.info(f"Using GPU {device_id}: {props.name}")
        logger.info(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"CUDA Capability: {props.major}.{props.minor}")
        logger.info(f"Number of SMs: {props.multi_processor_count}")
    except Exception:
        pass

def clear_gpu_memory():
    """Clear GPU memory cache"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def get_gpu_memory_info(device_id: Optional[int] = None) -> Dict[str, int]:
    """Get GPU memory usage information
    
    Args:
        device_id: Optional GPU device ID
        
    Returns:
        Dictionary containing memory information
    """
    if not torch.cuda.is_available():
        return {}
        
    try:
        if device_id is None:
            device_id = torch.cuda.current_device()
            
        return {
            'allocated': torch.cuda.memory_allocated(device_id),
            'cached': torch.cuda.memory_reserved(device_id),
            'total': torch.cuda.get_device_properties(device_id).total_memory
        }
    except Exception:
        return {}

def is_gpu_available() -> bool:
    """Check if GPU is available
    
    Returns:
        True if GPU is available
    """
    return torch.cuda.is_available()

def get_optimal_gpu() -> Optional[int]:
    """Get ID of GPU with most free memory
    
    Returns:
        GPU device ID or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None
        
    try:
        max_free = 0
        best_gpu = 0
        
        for i in range(torch.cuda.device_count()):
            memory_info = get_gpu_memory_info(i)
            free_memory = memory_info['total'] - memory_info['allocated']
            
            if free_memory > max_free:
                max_free = free_memory
                best_gpu = i
                
        return best_gpu
    except Exception:
        return 0 if torch.cuda.device_count() > 0 else None

# Export functions
__all__ = [
    'setup_gpu',
    'get_available_gpus',
    'clear_gpu_memory',
    'get_gpu_memory_info',
    'is_gpu_available',
    'get_optimal_gpu'
]