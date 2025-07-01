#!/usr/bin/env python3
"""
Safe training launcher with CUDA verification
"""

import torch
import subprocess
import sys

def check_cuda_requirement():
    """Check if CUDA is available before training"""
    print("🔍 Pre-training CUDA Check")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available!")
        print("\n🛑 PatchedBrainTransformer requires CUDA for training!")
        print("   Reasons:")
        print("   • Model is too large for CPU training")
        print("   • Training would take days/weeks on CPU")
        print("   • Mixed precision requires GPU")
        print("\n📋 To fix this:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA-enabled PyTorch:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Verify with: python verify_cuda.py")
        print("\n🛑 Training aborted - CUDA required!")
        return False
    
    print("✅ CUDA is available!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("   🚀 Ready for GPU training!")
    return True

def run_training():
    """Run the actual training script"""
    print("\n🚀 Starting PatchedBrainTransformer Training...")
    print("=" * 50)
    
    try:
        # Run the main training script
        result = subprocess.run([sys.executable, "main_pre_training.py"], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n🎉 Training completed successfully!")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running training: {e}")

def main():
    """Main function"""
    print("🧠 PatchedBrainTransformer - Safe Training Launcher")
    print("=" * 60)
    
    # Check CUDA first
    if not check_cuda_requirement():
        exit(1)
    
    # Run training if CUDA is available
    run_training()

if __name__ == "__main__":
    main()
