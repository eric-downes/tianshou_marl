"""Device utilities for cross-platform GPU support."""

import torch


def get_default_device():
    """Get the default device based on availability.
    
    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon/Metal) 
    3. CPU (fallback)
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def is_gpu_available():
    """Check if any GPU acceleration is available.
    
    Returns:
        bool: True if CUDA or MPS is available
    """
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def get_device_name():
    """Get a human-readable name for the current device.
    
    Returns:
        str: Device description
    """
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name()})"
    elif torch.backends.mps.is_available():
        return "Metal Performance Shaders (Apple Silicon)"
    else:
        return "CPU"


def validate_device(device: str) -> str:
    """Validate and normalize device string.
    
    Args:
        device: Requested device ('cuda', 'mps', 'cpu', 'auto')
        
    Returns:
        str: Valid device string
        
    Raises:
        ValueError: If requested device is not available
    """
    if device == "auto":
        return get_default_device()
    
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available. Use 'auto' or 'cpu'.")
    
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS (Metal) requested but not available. Use 'auto' or 'cpu'.")
    
    if device not in ["cuda", "mps", "cpu"]:
        raise ValueError(f"Unknown device: {device}. Use 'cuda', 'mps', 'cpu', or 'auto'.")
    
    return device