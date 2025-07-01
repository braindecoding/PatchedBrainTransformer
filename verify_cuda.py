#!/usr/bin/env python3
"""
Quick CUDA verification before training
"""

import torch

def verify_cuda():
    """Quick CUDA check"""
    print("🔍 CUDA Verification")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available!")
        print("\n🛑 PatchedBrainTransformer requires CUDA!")
        print("   Please install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    print("✅ CUDA is available!")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return True

if __name__ == "__main__":
    if verify_cuda():
        print("\n🚀 Ready for training!")
    else:
        print("\n🛑 Fix CUDA setup before training!")
        exit(1)
