"""
Configuration file for the Nunchaku Qwen-Image Interface.
Modify these settings to customize the interface behavior.
"""

import os
from pathlib import Path

# =============================================================================
# Model Configuration
# =============================================================================

# Base URL for model downloads
HUGGINGFACE_BASE_URL = "https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/resolve/main"

# Model configurations with download URLs
MODEL_CONFIGS = {
    "svdq-int4_r32": {
        "name": "SVDQuant INT4 Rank 32",
        "file": "svdq-int4_r32-qwen-image.safetensors",
        "description": "For non-Blackwell GPUs (pre-50-series). Fastest, lower quality.",
        "url": f"{HUGGINGFACE_BASE_URL}/svdq-int4_r32-qwen-image.safetensors"
    },
    "svdq-int4_r128": {
        "name": "SVDQuant INT4 Rank 128", 
        "file": "svdq-int4_r128-qwen-image.safetensors",
        "description": "For non-Blackwell GPUs (pre-50-series). Better quality, slower.",
        "url": f"{HUGGINGFACE_BASE_URL}/svdq-int4_r128-qwen-image.safetensors"
    },
    "svdq-fp4_r32": {
        "name": "SVDQuant FP4 Rank 32",
        "file": "svdq-fp4_r32-qwen-image.safetensors", 
        "description": "For Blackwell GPUs (50-series). Fastest, lower quality.",
        "url": f"{HUGGINGFACE_BASE_URL}/svdq-fp4_r32-qwen-image.safetensors"
    },
    "svdq-fp4_r128": {
        "name": "SVDQuant FP4 Rank 128",
        "file": "svdq-fp4_r128-qwen-image.safetensors",
        "description": "For Blackwell GPUs (50-series). Best quality, slowest.",
        "url": f"{HUGGINGFACE_BASE_URL}/svdq-fp4_r128-qwen-image.safetensors"
    }
}

# =============================================================================
# Interface Configuration
# =============================================================================

# Server settings
SERVER_HOST = "0.0.0.0"  # Bind to all interfaces
SERVER_PORT = 7860         # Default Gradio port

# Interface theme and appearance
INTERFACE_THEME = "soft"   # Gradio theme: "default", "soft", "glass", "monochrome"
INTERFACE_TITLE = "🎨 Nunchaku Qwen-Image Interface"

# =============================================================================
# Generation Parameters
# =============================================================================

# Default generation parameters
DEFAULT_INFERENCE_STEPS = 20
DEFAULT_CFG_VALUE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_SEED = -1  # -1 for random

# Parameter limits
MIN_INFERENCE_STEPS = 1
MAX_INFERENCE_STEPS = 100
MIN_CFG_VALUE = 1.0
MAX_CFG_VALUE = 20.0
MIN_IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 2048
IMAGE_SIZE_STEP = 64  # Must be multiples of this value

# =============================================================================
# File Paths
# =============================================================================

# Base directory (where this config file is located)
BASE_DIR = Path(__file__).parent

# Models directory (relative to base directory)
MODELS_DIR = BASE_DIR / "models"

# Logs directory
LOGS_DIR = BASE_DIR / "logs"

# Output directory for generated images
OUTPUT_DIR = BASE_DIR / "output"

# =============================================================================
# Logging Configuration
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Log file settings
LOG_TO_FILE = True
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

# =============================================================================
# Performance Configuration
# =============================================================================

# Device priority (first available will be used)
DEVICE_PRIORITY = ["cuda", "mps", "cpu"]

# Memory management
CLEAR_CUDA_CACHE_ON_UNLOAD = True
MAX_MEMORY_USAGE = 0.8  # Use up to 80% of available GPU memory

# Batch processing
ENABLE_BATCH_PROCESSING = False
MAX_BATCH_SIZE = 4

# =============================================================================
# Download Configuration
# =============================================================================

# Download settings
DOWNLOAD_CHUNK_SIZE = 8192
DOWNLOAD_TIMEOUT = 300  # 5 minutes
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY = 5  # seconds

# Progress bar settings
SHOW_DOWNLOAD_PROGRESS = True
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds

# =============================================================================
# Advanced Settings
# =============================================================================

# Enable experimental features
ENABLE_EXPERIMENTAL_FEATURES = False

# Model loading timeout
MODEL_LOADING_TIMEOUT = 300  # 5 minutes

# Image generation timeout
GENERATION_TIMEOUT = 600  # 10 minutes

# Enable model caching
ENABLE_MODEL_CACHING = True
MODEL_CACHE_SIZE = 2  # Number of models to keep in memory

# =============================================================================
# Validation Functions
# =============================================================================

def validate_image_size(width: int, height: int) -> bool:
    """Validate that image dimensions are within acceptable limits."""
    return (MIN_IMAGE_SIZE <= width <= MAX_IMAGE_SIZE and 
            MIN_IMAGE_SIZE <= height <= MAX_IMAGE_SIZE and
            width % IMAGE_SIZE_STEP == 0 and 
            height % IMAGE_SIZE_STEP == 0)

def validate_inference_steps(steps: int) -> bool:
    """Validate inference steps parameter."""
    return MIN_INFERENCE_STEPS <= steps <= MAX_INFERENCE_STEPS

def validate_cfg_value(cfg: float) -> bool:
    """Validate CFG value parameter."""
    return MIN_CFG_VALUE <= cfg <= MAX_CFG_VALUE

def get_model_info(model_key: str) -> dict:
    """Get information about a specific model."""
    if model_key not in MODEL_CONFIGS:
        return {"error": "Model not found"}
    
    config = MODEL_CONFIGS[model_key]
    model_path = MODELS_DIR / config["file"]
    
    return {
        "name": config["name"],
        "description": config["description"],
        "file": config["file"],
        "url": config["url"],
        "downloaded": model_path.exists(),
        "file_size": model_path.stat().st_size if model_path.exists() else 0
    }

# =============================================================================
# Environment Overrides
# =============================================================================

# Allow environment variables to override config
def get_config_value(key: str, default=None):
    """Get configuration value with environment variable override support."""
    env_key = f"NUNCHAKU_{key.upper()}"
    return os.getenv(env_key, globals().get(key, default))

# Apply environment overrides
if __name__ == "__main__":
    # Test configuration
    print("Configuration loaded successfully!")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
    print(f"Server port: {get_config_value('SERVER_PORT', SERVER_PORT)}")
