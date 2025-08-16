"""
Model loading and inference module for Qwen-Image models.
This module handles real image generation using the Qwen-Image model through diffusers.
"""

import torch
from pathlib import Path
import logging
from typing import Optional, Tuple
from PIL import Image
from config import MODEL_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verify SciPy availability before importing diffusers components. Diffusers
# schedulers such as DPMSolver depend on SciPy which might be missing in some
# environments (e.g. default Windows installs).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - best effort detection
    import scipy  # noqa: F401
    _scipy_available = True
except Exception as e:  # pragma: no cover - system-dependent
    _scipy_available = False
    logger.warning(
        "SciPy not available (%s). Install SciPy>=1.10 for DPMSolver support; "
        "falling back to EulerDiscreteScheduler.",
        e,
    )

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

            # Import diffusers components lazily after verifying SciPy availability
            from diffusers.utils import logging as diffusers_logging
            diffusers_logging.set_verbosity_info()

            if _scipy_available:
                from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler as _Scheduler
            else:
                from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler as _Scheduler
                logger.warning(
                    "SciPy missing; using EulerDiscreteScheduler. Install SciPy>=1.10 for DPMSolver support."
                )

            # FIX: The original code was loading the base model from Hugging Face instead of the
            # downloaded quantized model. This uses the local model file as intended.
            logger.info(f"Loading model from local file: {model_path}")

            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

            # Load the pipeline from the single quantized file
            try:
                if _scipy_available:
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                    )
                    # Prefer DPM++ 2M when SciPy is available
                    self.pipeline.scheduler = _Scheduler.from_config(
                        self.pipeline.scheduler.config
                    )
                else:
                    # Build Euler scheduler manually to avoid SciPy dependency
                    scheduler = _Scheduler(
                        beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule="scaled_linear",
                        timestep_spacing="leading",
                        steps_offset=1,
                    )
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        scheduler=scheduler,
                    )
                logger.info("Pipeline loaded successfully from single file.")
            except Exception as e:
                if not _scipy_available:
                    logger.error(
                        "Failed to load pipeline without SciPy: %s. Consider installing SciPy>=1.10.",
                        e,
                    )
                else:
                    logger.error(f"Failed to load pipeline from single file: {e}")
                return False

            # Optimize the pipeline (no scheduler configuration needed here if SciPy unavailable)

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
        if model_key not in MODEL_CONFIGS:
            return {"error": "Model not found"}
        model_path = self.models_dir / MODEL_CONFIGS[model_key]["file"]
        info = {
            "name": MODEL_CONFIGS[model_key]["name"],
            "description": MODEL_CONFIGS[model_key]["description"],
            "file": MODEL_CONFIGS[model_key]["file"],
            "downloaded": model_path.exists(),
            "file_size": model_path.stat().st_size if model_path.exists() else 0,
            "loaded": self.current_model == model_key and self.model_loaded
        }

        return info

    @property
    def available_models(self) -> dict:
        """Get dictionary of available models."""
        return MODEL_CONFIGS
