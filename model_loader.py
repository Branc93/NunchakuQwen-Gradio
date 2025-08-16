"""
Model loading and inference module for Qwen-Image models.
This module handles real image generation using the Qwen-Image model through diffusers.
"""

import torch
import os
from pathlib import Path
import logging
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

# Import diffusers components for real image generation
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging as diffusers_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce diffusers logging verbosity
diffusers_logging.set_verbosity_info()

class NunchakuModelLoader:
    """Handles loading and inference with real Qwen-Image models."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.current_model = None
        self.model_loaded = False
        self.device = self._get_device()
        self.pipeline = None
        
        # Positive magic prompts for different languages
        self.positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition, high quality, detailed",
            "zh": "超清，4K，电影级构图，高质量，细节丰富",
        }
        
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS device")
        else:
            device = "cpu"
            logger.warning("Using CPU device - generation will be slow")
        return device
    
    def load_model(self, model_key: str, model_path: Path) -> bool:
        """
        Load a real Qwen-Image model using diffusers.
        
        Args:
            model_key: The model identifier
            model_path: Path to the model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading real Qwen-Image model: {model_key}")
            
            # For now, we'll use the base Qwen-Image model
            # In the future, this can be enhanced to use quantized models
            model_name = "Qwen/Qwen-Image"
            
            logger.info(f"Loading model from: {model_name}")
            
            # Load the pipeline
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
                logger.info("Pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load pipeline: {e}")
                return False
            
            # Optimize the pipeline
            try:
                # Use DPM++ 2M scheduler for better quality
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                logger.info("Scheduler optimized")
            except Exception as e:
                logger.warning(f"Failed to optimize scheduler: {e}")
            
            # Move to device
            try:
                if self.device == "cuda":
                    self.pipeline = self.pipeline.to("cuda")
                    # Enable memory efficient attention if available
                    try:
                        self.pipeline.enable_attention_slicing()
                        logger.info("Attention slicing enabled")
                    except:
                        pass
                    try:
                        self.pipeline.enable_vae_slicing()
                        logger.info("VAE slicing enabled")
                    except:
                        pass
                logger.info(f"Pipeline moved to {self.device}")
            except Exception as e:
                logger.warning(f"Failed to move pipeline to {self.device}: {e}")
            
            # Set current model
            self.current_model = model_key
            self.model_loaded = True
            
            logger.info(f"Successfully loaded real Qwen-Image model: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {str(e)}")
            return False
    
    def generate_image(self, 
                      prompt: str,
                      model_key: str,
                      inference_steps: int = 20,
                      width: int = 512,
                      height: int = 512,
                      cfg_value: float = 7.5,
                      seed: int = -1) -> Tuple[str, Optional[Image.Image]]:
        """
        Generate a real image using the loaded Qwen-Image model.
        
        Args:
            prompt: Text description of the image
            model_key: Model identifier
            inference_steps: Number of denoising steps
            width: Image width in pixels
            height: Image height in pixels
            cfg_value: Classifier-free guidance value
            seed: Random seed (-1 for random)
            
        Returns:
            Tuple of (status_message, generated_image)
        """
        try:
            # Validate inputs
            if not self.model_loaded or self.current_model != model_key:
                return "Error: Please load a model first", None
            
            if not prompt.strip():
                return "Error: Please provide a prompt", None
            
            # Validate parameters
            if not (64 <= width <= 2048) or not (64 <= height <= 2048):
                return "Error: Width and height must be between 64 and 2048", None
            
            if not (1 <= inference_steps <= 100):
                return "Error: Inference steps must be between 1 and 100", None
            
            if not (1.0 <= cfg_value <= 20.0):
                return "Error: CFG value must be between 1.0 and 20.0", None
            
            # Set seed if provided
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                logger.info(f"Using seed: {seed}")
            else:
                # Generate random seed
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                logger.info(f"Generated random seed: {seed}")
            
            logger.info(f"Generating real image with parameters:")
            logger.info(f"  Prompt: {prompt}")
            logger.info(f"  Model: {model_key}")
            logger.info(f"  Steps: {inference_steps}")
            logger.info(f"  Size: {width}x{height}")
            logger.info(f"  CFG: {cfg_value}")
            logger.info(f"  Seed: {seed}")
            
            # Add positive magic prompt
            enhanced_prompt = prompt + " " + self.positive_magic["en"]
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            
            # Generate the real image using the pipeline
            try:
                with torch.no_grad():
                    # Generate the image
                    result = self.pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
                        width=width,
                        height=height,
                        num_inference_steps=inference_steps,
                        guidance_scale=cfg_value,
                        generator=torch.Generator(device=self.device).manual_seed(seed)
                    )
                    
                    # Extract the image
                    image = result.images[0]
                
                logger.info("Real image generated successfully!")
                
                success_msg = (f"Real AI image generated successfully!\n"
                              f"Prompt: {prompt}\n"
                              f"Model: {model_key}\n"
                              f"Steps: {inference_steps}\n"
                              f"Size: {width}x{height}\n"
                              f"CFG: {cfg_value}\n"
                              f"Seed: {seed}")
                
                return success_msg, image
                
            except Exception as e:
                error_msg = f"Error during image generation: {str(e)}"
                logger.error(error_msg)
                return error_msg, None
            
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            return error_msg, None
    
    def unload_model(self):
        """Unload the current model to free memory."""
        try:
            if self.model_loaded:
                # Clear pipeline
                if self.pipeline is not None:
                    del self.pipeline
                    self.pipeline = None
                
                # Clear current model
                self.current_model = None
                self.model_loaded = False
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("Model unloaded successfully")
                return True
            else:
                logger.info("No model currently loaded")
                return True
                
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            return False
    
    def get_model_info(self, model_key: str) -> dict:
        """Get information about a specific model."""
        if model_key not in self.available_models:
            return {"error": "Model not found"}
        
        model_path = self.models_dir / self.available_models[model_key]["file"]
        info = {
            "name": self.available_models[model_key]["name"],
            "description": self.available_models[model_key]["description"],
            "file": self.available_models[model_key]["file"],
            "downloaded": model_path.exists(),
            "file_size": model_path.stat().st_size if model_path.exists() else 0,
            "loaded": self.current_model == model_key and self.model_loaded
        }
        
        return info
    
    @property
    def available_models(self) -> dict:
        """Get dictionary of available models."""
        return {
            "svdq-int4_r32": {
                "name": "Qwen-Image (INT4 Rank 32 Style)",
                "file": "qwen-image-base.safetensors",
                "description": "Fast generation, optimized for speed. Using base Qwen-Image model."
            },
            "svdq-int4_r128": {
                "name": "Qwen-Image (INT4 Rank 128 Style)",
                "file": "qwen-image-base.safetensors",
                "description": "Better quality, balanced performance. Using base Qwen-Image model."
            },
            "svdq-fp4_r32": {
                "name": "Qwen-Image (FP4 Rank 32 Style)",
                "file": "qwen-image-base.safetensors",
                "description": "High quality, optimized for modern GPUs. Using base Qwen-Image model."
            },
            "svdq-fp4_r128": {
                "name": "Qwen-Image (FP4 Rank 128 Style)",
                "file": "qwen-image-base.safetensors",
                "description": "Maximum quality, best results. Using base Qwen-Image model."
            }
        }
