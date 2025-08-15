# 🚀 Quick Start Guide

## ⚡ Get Started in 3 Steps

### 1. **Install Dependencies**
```bash
# Windows (PowerShell) - RECOMMENDED
.\install.ps1

# Windows (Command Prompt)
install.bat

# Manual installation
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. **Launch the Interface**
```bash
python app.py
```

### 3. **Open Your Browser**
Navigate to: `http://localhost:7860`

---

## 🎯 What You'll See

- **4 Radio Buttons** for model selection:
  - `svdq-int4_r32` - Fastest, lower quality
  - `svdq-int4_r128` - Better quality, slower
  - `svdq-fp4_r32` - For Blackwell GPUs
  - `svdq-fp4_r128` - Best quality, slowest

- **Parameter Controls**:
  - Inference Steps: 1-100
  - Width/Height: 64-2048 pixels
  - CFG Value: 1.0-20.0
  - Seed: -1 for random

---

## 🔧 First Time Setup

1. **Download a Model**: Select your preferred model and click "📥 Download"
2. **Load the Model**: Click "🚀 Load Selected Model"
3. **Generate Images**: Enter a prompt and adjust parameters
4. **Click Generate**: Watch your image come to life!

---

## 💡 Pro Tips

- **Start with Rank 32** models for faster generation
- **Use Rank 128** for higher quality results
- **512x512** is a good starting resolution
- **20-30 steps** provides good quality/speed balance
- **CFG 7.5** is the sweet spot for most prompts

---

## 🆘 Need Help?

- **Test Installation**: `python test_installation.py`
- **Check GPU**: The interface will automatically detect your best device
- **Memory Issues**: Use smaller models or lower resolutions
- **Slow Generation**: Reduce inference steps or use Rank 32 models

---

## 🌟 Ready to Create?

Your Nunchaku Qwen-Image interface is ready! Start generating amazing images with the power of quantized AI models. 🎨✨
