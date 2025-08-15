# 🎨 Nunchaku Qwen-Image Interface

A modern Gradio web interface for generating high-quality images using quantized Qwen-Image models with SVDQuant technology. This interface provides an intuitive way to download, load, and use different quantized models based on your GPU type and quality preferences.

## ✨ Features

- **4 Model Variants**: Choose from different quantization levels and ranks
- **Automatic Downloads**: Download models directly from Hugging Face
- **Flexible Parameters**: Adjust inference steps, dimensions (up to 2048x2048), CFG value, and seed
- **Modern UI**: Clean, responsive interface built with Gradio
- **GPU Optimization**: Models optimized for different GPU generations

## 🚀 Model Options

### For Non-Blackwell GPUs (pre-50-series):
- **SVDQuant INT4 Rank 32**: Fastest generation, lower quality
- **SVDQuant INT4 Rank 128**: Better quality, slower generation

### For Blackwell GPUs (50-series):
- **SVDQuant FP4 Rank 32**: Fastest generation, lower quality  
- **SVDQuant FP4 Rank 128**: Best quality, slowest generation

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Windows 10/11

## 🛠️ Installation

1. **Clone or download this repository**
2. **Navigate to the project directory**:
   ```bash
   cd nunchaku-qwen-interface
   ```

3. **Quick Installation** (Windows):
   ```bash
   # PowerShell (recommended)
   .\install.ps1
   
   # Command Prompt
   install.bat
   ```

4. **Manual Installation**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Launch the Interface
```bash
python app.py
```

The interface will open in your browser at `http://localhost:7860`

### 2. Download a Model
1. Select your preferred model using the radio buttons
2. Click "📥 Download Selected Model"
3. Wait for the download to complete

### 3. Load the Model
1. Click "🚀 Load Selected Model"
2. Wait for the loading confirmation

### 4. Generate Images
1. Enter your image prompt
2. Adjust parameters as needed:
   - **Inference Steps**: 1-100 (higher = better quality, slower)
   - **Width/Height**: 64-2048 pixels (must be multiples of 64)
   - **CFG Value**: 1.0-20.0 (higher = more prompt adherence)
   - **Seed**: -1 for random, or specific number for reproducibility
3. Click "🎨 Generate Image"

## ⚙️ Parameter Guidelines

### Inference Steps
- **20-30**: Good balance of quality and speed
- **40-50**: Higher quality, slower generation
- **60+**: Maximum quality, slowest generation

### Image Dimensions
- **512x512**: Standard size, fast generation
- **1024x1024**: High resolution, good quality
- **2048x2048**: Maximum resolution, slowest generation

### CFG Value
- **7.0-8.0**: Balanced prompt adherence
- **10.0+**: Strong prompt adherence
- **5.0-6.0**: More creative, less prompt adherence

## 🔧 Technical Details

This interface uses:
- **Gradio**: Modern web interface framework
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models library
- **SVDQuant**: Advanced quantization technique (research paper)

**Note**: The actual Nunchaku inference engine and DeepCompressor quantization library are not yet publicly available. This interface provides a complete demo implementation that will be updated when those libraries become available.

## 📚 Research

Based on the paper: [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007)

## 🐛 Troubleshooting

### Common Issues:

1. **Model download fails**: Check internet connection and try again
2. **CUDA out of memory**: Reduce image dimensions or use a lower-rank model
3. **Slow generation**: Use fewer inference steps or a lower-rank model
4. **Model loading fails**: Ensure the model file was downloaded completely

### Performance Tips:

- Use Rank 32 models for faster generation
- Use Rank 128 models for better quality
- Match model type to your GPU generation
- Adjust batch size based on available VRAM

## 📄 License

This project is open source and available under the Apache 2.0 license.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For issues related to:
- **This interface**: Open an issue in this repository
- **Nunchaku models**: Visit [nunchaku.tech](https://nunchaku.tech)
- **SVDQuant research**: Check the [MIT HAN Lab](https://hanlab.mit.edu/)

---

**Happy Image Generation! 🎨✨**
