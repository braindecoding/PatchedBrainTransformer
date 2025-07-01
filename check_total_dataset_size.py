#!/usr/bin/env python3
"""
Check total available dataset size and analyze impact
"""

import sys
import numpy as np
import time
sys.path.append('src')

from utils import get_mindbigdata_eeg

def check_total_dataset_size():
    """Check how much data is actually available"""
    print("📊 Total Dataset Size Analysis")
    print("=" * 40)
    
    print("🔍 Loading FULL dataset (no limits)...")
    start_time = time.time()
    
    try:
        data, labels, meta, channels = get_mindbigdata_eeg(
            freq_min=0.5, freq_max=40, resample=250
        )
        
        load_time = time.time() - start_time
        
        if data is None:
            print("❌ Failed to load dataset")
            return None
        
        print(f"✅ Successfully loaded dataset in {load_time:.1f} seconds")
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total samples: {len(data):,}")
        print(f"   Data shape: {data.shape}")
        print(f"   Channels: {len(channels)} ({channels})")
        print(f"   Labels range: {min(labels)} - {max(labels)}")
        
        # Label distribution
        label_dist = np.bincount(labels)
        print(f"\n🔢 Label Distribution:")
        for digit, count in enumerate(label_dist):
            percentage = (count / len(labels)) * 100
            print(f"   Digit {digit}: {count:,} samples ({percentage:.1f}%)")
        
        # Check balance
        min_samples = min(label_dist)
        max_samples = max(label_dist)
        balance_ratio = min_samples / max_samples
        
        print(f"\n⚖️ Class Balance:")
        print(f"   Min samples per class: {min_samples:,}")
        print(f"   Max samples per class: {max_samples:,}")
        print(f"   Balance ratio: {balance_ratio:.3f}")
        
        if balance_ratio > 0.8:
            print("   ✅ Well balanced dataset")
        elif balance_ratio > 0.5:
            print("   ⚠️ Moderately balanced dataset")
        else:
            print("   ❌ Imbalanced dataset - consider class weighting")
        
        return len(data), data.shape, label_dist
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def analyze_training_impact(total_samples):
    """Analyze impact of using full dataset"""
    if total_samples is None:
        return
    
    print(f"\n🚀 Training Impact Analysis")
    print("=" * 30)
    
    # Training configuration
    test_size = 0.2
    batch_size = 32
    max_epochs = 200
    
    train_samples = int(total_samples * (1 - test_size))
    test_samples = total_samples - train_samples
    
    batches_per_epoch = train_samples // batch_size
    total_batches = batches_per_epoch * max_epochs
    
    print(f"📊 Training Configuration:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Train samples: {train_samples:,} ({(1-test_size)*100:.0f}%)")
    print(f"   Test samples: {test_samples:,} ({test_size*100:.0f}%)")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {batches_per_epoch:,}")
    print(f"   Max total batches: {total_batches:,}")
    
    # Time estimation (rough)
    # Assume ~0.5 seconds per batch on GPU (conservative estimate)
    seconds_per_batch = 0.5
    estimated_time_per_epoch = batches_per_epoch * seconds_per_batch
    estimated_max_time = estimated_time_per_epoch * max_epochs
    
    print(f"\n⏱️ Time Estimation:")
    print(f"   Est. time per epoch: {estimated_time_per_epoch/60:.1f} minutes")
    print(f"   Est. max training time: {estimated_max_time/3600:.1f} hours")
    
    # Memory estimation
    # Rough estimate: 4 bytes per float32 * samples * channels * timepoints
    channels = 14
    timepoints = 256
    bytes_per_sample = 4 * channels * timepoints
    total_memory_gb = (total_samples * bytes_per_sample) / (1024**3)
    
    print(f"\n💾 Memory Estimation:")
    print(f"   Est. dataset memory: {total_memory_gb:.1f} GB")
    
    if total_memory_gb < 8:
        print("   ✅ Should fit in GPU memory")
    elif total_memory_gb < 16:
        print("   ⚠️ May need memory optimization")
    else:
        print("   ❌ May require data streaming")
    
    # Performance prediction
    samples_per_param = estimate_samples_per_parameter(train_samples)
    
    print(f"\n🎯 Performance Prediction:")
    print(f"   Samples per parameter: {samples_per_param:.1f}")
    
    if samples_per_param > 20:
        print("   ✅ Excellent data/model ratio - expect great performance")
    elif samples_per_param > 10:
        print("   ✅ Good data/model ratio - expect good performance")
    elif samples_per_param > 5:
        print("   ⚠️ Moderate ratio - watch for overfitting")
    else:
        print("   ❌ Low ratio - high overfitting risk")

def estimate_samples_per_parameter(train_samples):
    """Estimate samples per model parameter"""
    # Model configuration (from main_pre_training.py)
    d_model = 64
    num_heads = 4
    num_blocks = 2
    d_ff = 64 * 2
    
    # Rough parameter estimation
    params_per_block = (
        4 * d_model * d_model +  # Q, K, V, O projections
        2 * d_model * d_ff +     # FFN layers
        4 * d_model              # Layer norms and biases
    )
    
    total_params = (
        num_blocks * params_per_block +  # Transformer blocks
        d_model * 10 +                   # Classification head
        1000                             # Embeddings and misc
    )
    
    return train_samples / total_params

def compare_dataset_sizes():
    """Compare different dataset size scenarios"""
    print(f"\n📊 Dataset Size Comparison")
    print("=" * 30)
    
    scenarios = [
        ("Previous (5K)", 5000),
        ("Current (20K)", 20000),
        ("Full Dataset", None)  # Will be filled with actual size
    ]
    
    # Get actual full dataset size
    total_samples, _, _ = check_total_dataset_size()
    if total_samples:
        scenarios[2] = ("Full Dataset", total_samples)
    
    print(f"{'Scenario':<15} {'Samples':<10} {'Train':<8} {'Test':<8} {'Overfitting Risk'}")
    print("-" * 65)
    
    for name, samples in scenarios:
        if samples is None:
            continue
            
        train = int(samples * 0.8)
        test = int(samples * 0.2)
        
        # Risk assessment based on samples per parameter
        spp = estimate_samples_per_parameter(train)
        if spp > 15:
            risk = "LOW"
        elif spp > 8:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        print(f"{name:<15} {samples:<10,} {train:<8,} {test:<8,} {risk}")

def main():
    """Main analysis function"""
    print("🧠 PatchedBrainTransformer - Total Dataset Analysis")
    print("=" * 60)
    
    # Check total dataset size
    result = check_total_dataset_size()
    
    if result:
        total_samples, shape, distribution = result
        
        # Analyze training impact
        analyze_training_impact(total_samples)
        
        # Compare scenarios
        compare_dataset_sizes()
        
        print(f"\n🎯 RECOMMENDATION:")
        print("=" * 20)
        
        if total_samples > 50000:
            print("✅ EXCELLENT! Use full dataset for maximum performance")
            print("   Large dataset will significantly improve generalization")
        elif total_samples > 30000:
            print("✅ VERY GOOD! Use full dataset")
            print("   Good dataset size for robust training")
        elif total_samples > 20000:
            print("✅ GOOD! Use full dataset")
            print("   Better than current 20K limit")
        else:
            print("⚠️ Consider keeping current 20K limit")
            print("   Full dataset may not provide significant benefit")
        
        print(f"\n🚀 Next Steps:")
        print("   1. Run training with full dataset")
        print("   2. Monitor GPU memory usage")
        print("   3. Compare results with 20K limited version")
        print("   4. Adjust batch size if memory issues occur")

if __name__ == "__main__":
    main()
