import gradio as gr
import torch
import os
from pathlib import Path
import hashlib
import requests
from tqdm import tqdm
import time
from model_loader import NunchakuModelLoader

# Model configurations
MODEL_CONFIGS = {
    "svdq-int4_r32": {
        "name": "SVDQuant INT4 Rank 32",
        "file": "svdq-int4_r32-qwen-image.safetensors",
        "description": "For non-Blackwell GPUs (pre-50-series). Fastest, lower quality.",
        "url": "https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/resolve/main/svdq-int4_r32-qwen-image.safetensors"
    },
    "svdq-int4_r128": {
        "name": "SVDQuant INT4 Rank 128", 
        "file": "svdq-int4_r128-qwen-image.safetensors",
        "description": "For non-Blackwell GPUs (pre-50-series). Better quality, slower.",
        "url": "https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/resolve/main/svdq-int4_r128-qwen-image.safetensors"
    },
    "svdq-fp4_r32": {
        "name": "SVDQuant FP4 Rank 32",
        "file": "svdq-fp4_r32-qwen-image.safetensors", 
        "description": "For Blackwell GPUs (50-series). Fastest, lower quality.",
        "url": "https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/resolve/main/svdq-fp4_r32-qwen-image.safetensors"
    },
    "svdq-fp4_r128": {
        "name": "SVDQuant FP4 Rank 128",
        "file": "svdq-fp4_r128-qwen-image.safetensors",
        "description": "For Blackwell GPUs (50-series). Best quality, slowest.",
        "url": "https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/resolve/main/svdq-fp4_r128-qwen-image.safetensors"
    }
}

class NunchakuQwenInterface:
    def __init__(self):
        self.model_loader = NunchakuModelLoader()
        self.models_dir = self.model_loader.models_dir
        
    def download_model(self, model_key, progress=gr.Progress()):
        """Download the selected model file"""
        if model_key not in MODEL_CONFIGS:
            return f"Error: Invalid model key {model_key}"
            
        config = MODEL_CONFIGS[model_key]
        model_path = self.models_dir / config["file"]
        
        # Check if model already exists
        if model_path.exists():
            return f"Model {config['name']} already exists at {model_path}"
        
        # Download the model
        try:
            progress(0, desc=f"Downloading {config['name']}...")
            
            response = requests.get(config["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress(downloaded / total_size, desc=f"Downloading {config['name']}...")
            
            progress(1.0, desc=f"Download complete!")
            return f"Successfully downloaded {config['name']} to {model_path}"
            
        except Exception as e:
            return f"Error downloading model: {str(e)}"
    
    def load_model(self, model_key):
        """Load the selected model"""
        if model_key not in MODEL_CONFIGS:
            return "Error: Invalid model key"
            
        config = MODEL_CONFIGS[model_key]
        model_path = self.models_dir / config["file"]
        
        if not model_path.exists():
            return f"Model file not found. Please download {config['name']} first."
        
        try:
            # Use the model loader to load the model
            success = self.model_loader.load_model(model_key, model_path)
            
            if success:
                return f"Successfully loaded {config['name']}"
            else:
                return f"Failed to load {config['name']}"
            
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def generate_image(self, prompt, model_key, inference_steps, width, height, cfg_value, seed):
        """Generate image using the selected model and parameters"""
        try:
            # Use the model loader to generate the image
            status_msg, generated_image = self.model_loader.generate_image(
                prompt=prompt,
                model_key=model_key,
                inference_steps=inference_steps,
                width=width,
                height=height,
                cfg_value=cfg_value,
                seed=seed
            )
            
            return status_msg, generated_image
            
        except Exception as e:
            return f"Error generating image: {str(e)}", None
    
    def unload_model(self):
        """Unload the current model to free memory"""
        try:
            success = self.model_loader.unload_model()
            if success:
                return "Model unloaded successfully"
            else:
                return "Failed to unload model"
        except Exception as e:
            return f"Error unloading model: {str(e)}"

def create_interface():
    interface = NunchakuQwenInterface()
    
    with gr.Blocks(title="Nunchaku Qwen-Image Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Nunchaku Qwen-Image Interface")
        gr.Markdown("Generate high-quality images using quantized Qwen-Image models with SVDQuant technology.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📥 Model Management")
                
                # Model selection radio buttons
                model_radio = gr.Radio(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="svdq-int4_r32",
                    label="Select Model",
                    info="Choose the model based on your GPU type and quality preferences"
                )
                
                # Model info display
                model_info = gr.Markdown()
                
                # Model status indicator
                model_status = gr.Textbox(
                    label="Current Model Status",
                    value="No model loaded",
                    interactive=False
                )
                
                # Download button
                download_btn = gr.Button("📥 Download Selected Model", variant="primary")
                download_status = gr.Textbox(label="Download Status", interactive=False)
                
                # Load button
                load_btn = gr.Button("🚀 Load Selected Model", variant="secondary")
                load_status = gr.Textbox(label="Load Status", interactive=False)
                
                # Unload button
                unload_btn = gr.Button("🗑️ Unload Current Model", variant="stop")
                unload_status = gr.Textbox(label="Unload Status", interactive=False)
                
                # Update model info when selection changes
                def update_model_info(model_key):
                    if model_key in MODEL_CONFIGS:
                        config = MODEL_CONFIGS[model_key]
                        info = f"""
                        **{config['name']}**
                        
                        {config['description']}
                        
                        **File:** `{config['file']}`
                        """
                        return info
                    return "Please select a model"
                
                # Update model status
                def update_model_status():
                    if interface.model_loader.model_loaded:
                        current_model = interface.model_loader.current_model
                        if current_model in MODEL_CONFIGS:
                            return f"✅ Loaded: {MODEL_CONFIGS[current_model]['name']}"
                        else:
                            return f"✅ Loaded: {current_model}"
                    else:
                        return "❌ No model loaded"
                
                model_radio.change(
                    fn=update_model_info,
                    inputs=[model_radio],
                    outputs=[model_info]
                )
                
                # Download functionality
                download_btn.click(
                    fn=interface.download_model,
                    inputs=[model_radio],
                    outputs=[download_status]
                )
                
                # Load functionality
                load_btn.click(
                    fn=interface.load_model,
                    inputs=[model_radio],
                    outputs=[load_status]
                ).then(
                    fn=update_model_status,
                    outputs=[model_status]
                )
                
                # Unload functionality
                unload_btn.click(
                    fn=interface.unload_model,
                    outputs=[unload_status]
                ).then(
                    fn=update_model_status,
                    outputs=[model_status]
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 Image Generation")
                
                # Prompt input
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column():
                        # Model selection for generation
                        gen_model_radio = gr.Radio(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="svdq-int4_r32",
                            label="Model for Generation"
                        )
                        
                        # Inference parameters
                        inference_steps = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1,
                            label="Inference Steps",
                            info="Higher values = better quality, slower generation"
                        )
                        
                        cfg_value = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="CFG Value",
                            info="Higher values = more prompt adherence"
                        )
                    
                    with gr.Column():
                        # Image dimensions
                        width = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Width (pixels)"
                        )
                        
                        height = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Height (pixels)"
                        )
                        
                        seed = gr.Number(
                            value=-1,
                            label="Seed",
                            info="-1 for random seed"
                        )
                
                # Generate button
                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                
                # Output
                output_text = gr.Textbox(label="Generation Status", interactive=False)
                output_image = gr.Image(label="Generated Image")
                
                # Generate functionality
                generate_btn.click(
                    fn=interface.generate_image,
                    inputs=[prompt_input, gen_model_radio, inference_steps, width, height, cfg_value, seed],
                    outputs=[output_text, output_image]
                )
        
        # Initialize model info and status
        demo.load(lambda: update_model_info("svdq-int4_r32"), outputs=[model_info])
        demo.load(lambda: update_model_status(), outputs=[model_status])
        
        gr.Markdown("---")
        gr.Markdown("""
        ## 📚 About
        
        This interface uses **Nunchaku** quantized versions of **Qwen-Image**, designed to generate high-quality images from text prompts with advances in complex text rendering.
        
        **Model Types:**
        - **INT4 models**: For non-Blackwell GPUs (pre-50-series)
        - **FP4 models**: For Blackwell GPUs (50-series)
        - **Rank 32**: Faster generation, lower quality
        - **Rank 128**: Slower generation, better quality
        
        **Paper:** [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007)
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
