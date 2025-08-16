#!/usr/bin/env python3
"""
Check what Nunchaku components are available.
"""

import sys

def check_nunchaku_availability():
    """Check what Nunchaku components are available."""
    print("🔍 Checking Nunchaku Availability...")
    print("=" * 50)
    
    # Check diffusers
    try:
        import diffusers
        print(f"✅ Diffusers imported: {diffusers.__version__}")
        
        # Check if nunchaku components are available
        try:
            from diffusers.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
            print("✅ NunchakuQwenImageTransformer2DModel found in diffusers")
        except ImportError:
            print("❌ NunchakuQwenImageTransformer2DModel not found in diffusers")
        
        try:
            from diffusers.pipeline.pipeline_qwenimage import NunchakuQwenImagePipeline
            print("✅ NunchakuQwenImagePipeline found in diffusers")
        except ImportError:
            print("❌ NunchakuQwenImagePipeline not found in diffusers")
            
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    # Check if we can find nunchaku in other locations
    print("\n🔍 Searching for Nunchaku components...")
    
    # Try to find nunchaku in the current environment
    try:
        import nunchaku
        print(f"✅ Found nunchaku package: {nunchaku.__file__}")
        print(f"   Contents: {dir(nunchaku)}")
    except ImportError:
        print("❌ No nunchaku package found")
    
    return True

if __name__ == "__main__":
    check_nunchaku_availability()
