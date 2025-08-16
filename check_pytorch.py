#!/usr/bin/env python3
"""
Check PyTorch installation and CUDA availability.
This script helps verify that the CUDA version of PyTorch is working.
"""

import sys

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print("🔍 Checking PyTorch Installation...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        print(f"   Build: {torch.version.git_version}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA is available!")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Device: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test CUDA tensor creation
            try:
                x = torch.randn(3, 3).cuda()
                print(f"✅ CUDA tensor creation successful")
                print(f"   Tensor device: {x.device}")
                print(f"   Tensor shape: {x.shape}")
            except Exception as e:
                print(f"❌ CUDA tensor creation failed: {e}")
                return False
                
        else:
            print("❌ CUDA is NOT available")
            print("   This might mean PyTorch CPU version was installed")
            print("   Please run install.ps1 or install.bat to get CUDA version")
            return False
            
        # Check if this is the CUDA version
        if "cu" in torch.__version__.lower():
            print(f"✅ This appears to be a CUDA build of PyTorch")
        else:
            print(f"⚠️  This might be a CPU build of PyTorch")
            print(f"   Version string doesn't contain 'cu'")
            
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_other_packages():
    """Check other important packages."""
    print("\n🔍 Checking Other Packages...")
    print("=" * 50)
    
    packages = [
        ("gradio", "Gradio UI framework"),
        ("diffusers", "Hugging Face diffusion models"),
        ("transformers", "Hugging Face transformers"),
        ("PIL", "Pillow image processing"),
        ("numpy", "Numerical computing"),
    ]
    
    all_good = True
    for package, description in packages:
        try:
            if package == "PIL":
                import PIL
                print(f"✅ {package}: {description} (version: {PIL.__version__})")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {package}: {description} (version: {version})")
        except ImportError:
            print(f"❌ {package}: {description} - Not installed")
            all_good = False
        except Exception as e:
            print(f"⚠️  {package}: {description} - Error: {e}")
            all_good = False
    
    return all_good

def main():
    """Run all checks."""
    print("🚀 PyTorch CUDA Checker")
    print("=" * 50)
    
    # Check PyTorch
    pytorch_ok = check_pytorch()
    
    # Check other packages
    packages_ok = check_other_packages()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"   PyTorch CUDA: {'✅ Working' if pytorch_ok else '❌ Not Working'}")
    print(f"   Other Packages: {'✅ All Good' if packages_ok else '⚠️  Some Issues'}")
    
    if pytorch_ok and packages_ok:
        print("\n🎉 Everything looks good! You can run the interface.")
        print("   Run: python app.py")
    elif not pytorch_ok:
        print("\n❌ PyTorch CUDA is not working properly.")
        print("   Please run: .\\install.ps1")
    else:
        print("\n⚠️  Some packages have issues. Please check the errors above.")
    
    return pytorch_ok and packages_ok

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
