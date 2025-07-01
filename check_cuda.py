#!/usr/bin/env python3
"""
Script untuk mengecek CUDA setup dan optimasi GPU
"""

import torch
import sys

def check_cuda_setup():
    """Comprehensive CUDA setup check"""
    print("🔍 CUDA Setup Check")
    print("=" * 50)
    
    # Basic CUDA availability
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   🛑 PatchedBrainTransformer requires CUDA for training!")
        print("   Please ensure:")
        print("   1. NVIDIA GPU is installed and drivers are up to date")
        print("   2. Install CUDA-enabled PyTorch:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Verify NVIDIA drivers: nvidia-smi")
        print("\n   ⚠️ Training will NOT start without CUDA!")
        return False
    
    # CUDA details
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
    
    # GPU details
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n🚀 GPU {i}: {props.name}")
        print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Multiprocessors: {props.multi_processor_count}")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n🎯 Current Device: {current_device}")
    print(f"   Device Name: {torch.cuda.get_device_name(current_device)}")
    
    # Memory check
    print(f"\n💾 Memory Status:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"   Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Optimization settings
    print(f"\n⚡ Optimization Settings:")
    print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   TF32 (cuDNN): {torch.backends.cudnn.allow_tf32}")
    
    return True

def test_gpu_performance():
    """Test basic GPU performance"""
    print("\n🏃 GPU Performance Test")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for performance test")
        return
    
    device = torch.device('cuda')
    
    # Test tensor operations
    print("Testing tensor operations...")
    
    import time
    
    # Matrix multiplication test
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up
    for _ in range(5):
        _ = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(10):
        c = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    gflops = (2 * size**3) / (avg_time * 1e9)
    
    print(f"✅ Matrix Multiplication ({size}x{size}):")
    print(f"   Average time: {avg_time*1000:.2f} ms")
    print(f"   Performance: {gflops:.1f} GFLOPS")
    
    # Memory bandwidth test
    print("\nTesting memory bandwidth...")
    
    size_mb = 100
    elements = size_mb * 1024 * 1024 // 4  # 4 bytes per float32
    data = torch.randn(elements, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(10):
        _ = data * 2.0
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    bandwidth = (size_mb * 2) / avg_time  # Read + Write
    
    print(f"✅ Memory Bandwidth Test ({size_mb} MB):")
    print(f"   Average time: {avg_time*1000:.2f} ms")
    print(f"   Bandwidth: {bandwidth:.1f} MB/s")

def optimize_cuda_settings():
    """Apply optimal CUDA settings"""
    print("\n⚙️ Applying CUDA Optimizations")
    print("=" * 35)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for optimization")
        return
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ Enabled cuDNN benchmark mode")
    print("✅ Enabled TF32 for matrix operations")
    print("✅ Enabled TF32 for cuDNN operations")
    
    # Clear cache
    torch.cuda.empty_cache()
    print("✅ Cleared GPU cache")
    
    print("\n🎉 CUDA optimizations applied!")

def main():
    """Main function"""
    print("🧠 PatchedBrainTransformer - CUDA Setup Checker")
    print("=" * 60)
    
    # Check CUDA setup
    cuda_available = check_cuda_setup()
    
    if cuda_available:
        # Test performance
        test_gpu_performance()
        
        # Apply optimizations
        optimize_cuda_settings()
        
        print("\n✅ CUDA setup is ready for training!")
        print("   You can now run: python main_pre_training.py")
    else:
        print("\n❌ Please fix CUDA setup before training")
        print("   Install CUDA-enabled PyTorch and try again")

if __name__ == "__main__":
    main()
