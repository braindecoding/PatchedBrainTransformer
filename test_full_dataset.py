#!/usr/bin/env python3
"""
Test loading and training with full dataset
"""

import sys
import time
import torch
import numpy as np
sys.path.append('src')

from utils import get_mindbigdata_eeg, train_test_split

def test_full_dataset_loading():
    """Test loading full dataset"""
    print("🧪 Testing Full Dataset Loading")
    print("=" * 35)
    
    print("📊 Loading full dataset...")
    start_time = time.time()
    
    try:
        data, labels, meta, channels = get_mindbigdata_eeg(
            freq_min=0.5, freq_max=40, resample=250
        )
        
        load_time = time.time() - start_time
        
        if data is None:
            print("❌ Failed to load dataset")
            return False
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Load time: {load_time:.1f} seconds")
        print(f"   Total samples: {len(data):,}")
        print(f"   Data shape: {data.shape}")
        print(f"   Memory usage: ~{data.nbytes / 1024**3:.1f} GB")
        
        # Test train/test split
        print(f"\n🔄 Testing train/test split...")
        split_start = time.time()
        
        train_data, train_labels, train_meta, test_data, test_labels, test_meta = train_test_split(
            data, labels, meta, test_size=0.2
        )
        
        split_time = time.time() - split_start
        
        print(f"✅ Split completed in {split_time:.1f} seconds")
        print(f"   Train samples: {len(train_data):,}")
        print(f"   Test samples: {len(test_data):,}")
        print(f"   Train memory: ~{train_data.nbytes / 1024**3:.1f} GB")
        print(f"   Test memory: ~{test_data.nbytes / 1024**3:.1f} GB")
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            print(f"\n🚀 GPU Memory Check:")
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU Total Memory: {gpu_total:.1f} GB")
            
            # Estimate if data fits in GPU
            estimated_gpu_usage = (train_data.nbytes + test_data.nbytes) / 1024**3
            print(f"   Estimated Data Usage: {estimated_gpu_usage:.1f} GB")
            
            if estimated_gpu_usage < gpu_total * 0.7:
                print("   ✅ Should fit in GPU memory")
            else:
                print("   ⚠️ May need memory optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency with different batch sizes"""
    print(f"\n💾 Memory Efficiency Test")
    print("=" * 25)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - skipping GPU memory test")
        return
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]
    
    print("Testing batch sizes for memory usage...")
    
    for batch_size in batch_sizes:
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Create dummy data similar to your model input
            # Assuming: batch_size x sequence_length x d_model
            dummy_input = torch.randn(batch_size, 100, 64, device='cuda')
            dummy_target = torch.randint(0, 10, (batch_size,), device='cuda')
            
            # Measure memory
            memory_before = torch.cuda.memory_allocated() / 1024**3
            
            # Simulate forward pass
            dummy_output = torch.nn.Linear(64, 10, device='cuda')(dummy_input.mean(dim=1))
            loss = torch.nn.CrossEntropyLoss()(dummy_output, dummy_target)
            
            memory_after = torch.cuda.memory_allocated() / 1024**3
            memory_used = memory_after - memory_before
            
            print(f"   Batch size {batch_size:3d}: {memory_used:.3f} GB")
            
            # Clean up
            del dummy_input, dummy_target, dummy_output, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch size {batch_size:3d}: ❌ Out of memory")
            else:
                print(f"   Batch size {batch_size:3d}: ❌ Error: {e}")

def estimate_training_time():
    """Estimate training time with full dataset"""
    print(f"\n⏱️ Training Time Estimation")
    print("=" * 30)
    
    # Load dataset to get actual size
    try:
        data, labels, meta, channels = get_mindbigdata_eeg()
        if data is None:
            print("❌ Cannot estimate - dataset not loaded")
            return
        
        total_samples = len(data)
        train_samples = int(total_samples * 0.8)
        
    except:
        print("⚠️ Using estimated dataset size")
        total_samples = 50000  # Conservative estimate
        train_samples = 40000
    
    # Training parameters
    batch_size = 64  # Updated batch size
    max_epochs = 300  # Updated max epochs
    early_stopping_patience = 100
    
    batches_per_epoch = train_samples // batch_size
    
    print(f"📊 Training Configuration:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Train samples: {train_samples:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {batches_per_epoch:,}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    
    # Time estimation scenarios
    scenarios = [
        ("Fast convergence (50 epochs)", 50),
        ("Normal convergence (150 epochs)", 150),
        ("Slow convergence (250 epochs)", 250),
        ("Max epochs (300 epochs)", 300)
    ]
    
    print(f"\n⏱️ Time Estimates (assuming 0.5s per batch):")
    
    for scenario_name, epochs in scenarios:
        total_batches = batches_per_epoch * epochs
        estimated_seconds = total_batches * 0.5
        estimated_hours = estimated_seconds / 3600
        
        print(f"   {scenario_name}: {estimated_hours:.1f} hours")

def recommend_configuration():
    """Recommend optimal configuration for full dataset"""
    print(f"\n💡 Configuration Recommendations")
    print("=" * 35)
    
    print("✅ Optimizations already applied:")
    print("   • Unlimited dataset loading (max_trials = None)")
    print("   • Increased batch size: 32 → 64")
    print("   • Increased max epochs: 200 → 300")
    print("   • Increased learning rate: 5e-5 → 1e-4")
    print("   • Increased warmup: 50 → 100 iterations")
    print("   • Early stopping patience: 100 epochs")
    
    print(f"\n🎯 Additional recommendations:")
    print("   • Monitor GPU memory usage during training")
    print("   • Consider gradient accumulation if memory issues")
    print("   • Use mixed precision (already enabled)")
    print("   • Save checkpoints more frequently (every 10-20 epochs)")
    
    print(f"\n⚠️ Watch out for:")
    print("   • Longer training time (potentially 4-8 hours)")
    print("   • Higher GPU memory usage")
    print("   • Need for larger storage for checkpoints")

def main():
    """Main test function"""
    print("🧠 PatchedBrainTransformer - Full Dataset Test")
    print("=" * 55)
    
    # Test dataset loading
    success = test_full_dataset_loading()
    
    if success:
        # Test memory efficiency
        test_memory_efficiency()
        
        # Estimate training time
        estimate_training_time()
        
        # Provide recommendations
        recommend_configuration()
        
        print(f"\n🎉 READY FOR FULL DATASET TRAINING!")
        print("   Run: python main_pre_training.py")
    else:
        print(f"\n❌ Issues detected. Please fix before training.")

if __name__ == "__main__":
    main()
