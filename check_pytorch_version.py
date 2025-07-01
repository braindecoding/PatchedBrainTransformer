#!/usr/bin/env python3
"""
Check PyTorch version and AMP compatibility
"""

import torch

def check_pytorch_amp():
    """Check PyTorch version and AMP support"""
    print("🔍 PyTorch & AMP Compatibility Check")
    print("=" * 45)
    
    # Basic PyTorch info
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Check AMP imports
    print("\n🔍 AMP Import Check:")
    
    # Try new import path (PyTorch 2.0+)
    try:
        from torch.amp import autocast, GradScaler
        print("✅ torch.amp import: SUCCESS (PyTorch 2.0+ style)")
        amp_available = True
        amp_source = "torch.amp"
    except ImportError:
        print("❌ torch.amp import: FAILED")
        
        # Try old import path (PyTorch 1.6+)
        try:
            from torch.cuda.amp import autocast, GradScaler
            print("✅ torch.cuda.amp import: SUCCESS (PyTorch 1.6+ style)")
            amp_available = True
            amp_source = "torch.cuda.amp"
        except ImportError:
            print("❌ torch.cuda.amp import: FAILED")
            amp_available = False
            amp_source = None
    
    # Test AMP functionality
    if amp_available and torch.cuda.is_available():
        print(f"\n🧪 Testing AMP with {amp_source}:")
        try:
            if amp_source == "torch.amp":
                from torch.amp import autocast, GradScaler
                scaler = GradScaler('cuda')
                autocast_ctx = autocast('cuda')
            else:
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
                autocast_ctx = autocast()
            
            # Simple test with gradient requirement
            device = torch.device('cuda')
            x = torch.randn(2, 3, device=device, requires_grad=True)

            with autocast_ctx:
                y = x * 2.0

            loss = y.sum()
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD([x], lr=0.1))
            scaler.update()
            
            print("✅ AMP functionality: WORKING")
            
        except Exception as e:
            print(f"❌ AMP functionality: FAILED - {e}")
            amp_available = False
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
    print(f"   AMP: {'✅' if amp_available else '❌'}")
    
    if amp_available:
        print(f"   AMP Source: {amp_source}")
        print("   🚀 Ready for mixed precision training!")
    else:
        print("   ⚠️ AMP not available - will use standard precision")
    
    return amp_available

def recommend_pytorch_version():
    """Recommend PyTorch version if needed"""
    current_version = torch.__version__
    major, minor = map(int, current_version.split('.')[:2])
    
    print(f"\n💡 PyTorch Version Analysis:")
    
    if major >= 2:
        print("✅ PyTorch 2.0+ detected - Latest features available")
    elif major == 1 and minor >= 6:
        print("✅ PyTorch 1.6+ detected - AMP supported")
    else:
        print("⚠️ Old PyTorch version detected")
        print("   Recommended: Upgrade to PyTorch 2.0+")
        print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def main():
    """Main function"""
    print("🧠 PatchedBrainTransformer - PyTorch Compatibility Check")
    print("=" * 65)
    
    amp_available = check_pytorch_amp()
    recommend_pytorch_version()
    
    if torch.cuda.is_available() and amp_available:
        print("\n🎉 System is ready for optimal training!")
    elif torch.cuda.is_available():
        print("\n⚠️ CUDA available but AMP may have issues")
    else:
        print("\n❌ CUDA not available - training will not start")

if __name__ == "__main__":
    main()
