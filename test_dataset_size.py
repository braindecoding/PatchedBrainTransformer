#!/usr/bin/env python3
"""
Test dataset size and overfitting prevention measures
"""

import sys
sys.path.append('src')

from utils import get_mindbigdata_eeg, train_test_split
import numpy as np

def test_dataset_sizes():
    """Test different dataset sizes"""
    print("🔍 Testing Dataset Sizes for Overfitting Analysis")
    print("=" * 60)
    
    try:
        # Load full dataset
        print("📊 Loading MindBigData dataset...")
        data, labels, meta, channels = get_mindbigdata_eeg(
            freq_min=0.5, freq_max=40, resample=250
        )
        
        if data is None:
            print("❌ Failed to load dataset")
            return
        
        print(f"✅ Total samples loaded: {len(data)}")
        print(f"📈 Data shape: {data.shape}")
        print(f"🔢 Label distribution: {np.bincount(labels)}")
        
        # Test different train/test splits
        test_sizes = [0.05, 0.1, 0.2, 0.3]
        
        print(f"\n📋 Train/Test Split Analysis:")
        print("=" * 40)
        
        for test_size in test_sizes:
            train_data, train_labels, train_meta, test_data, test_labels, test_meta = train_test_split(
                data, labels, meta, test_size=test_size
            )
            
            print(f"\nTest Size: {test_size*100:.0f}%")
            print(f"  Train samples: {len(train_data)}")
            print(f"  Test samples: {len(test_data)}")
            print(f"  Train/Test ratio: {len(train_data)/len(test_data):.1f}:1")
            
            # Check class balance
            train_dist = np.bincount(train_labels)
            test_dist = np.bincount(test_labels)
            print(f"  Train distribution: {train_dist}")
            print(f"  Test distribution: {test_dist}")
        
        # Recommend optimal split
        print(f"\n💡 Recommendations:")
        print("=" * 20)
        print("✅ Use 20% test split for better validation")
        print("✅ Increased dataset size to 20,000 samples")
        print("✅ Enhanced regularization (dropout=0.2, weight_decay=0.05)")
        print("✅ Reduced model complexity (2 blocks, 4 heads)")
        print("✅ Early stopping with patience=20")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def analyze_overfitting_risk():
    """Analyze overfitting risk factors"""
    print(f"\n🔍 Overfitting Risk Analysis")
    print("=" * 30)
    
    # Model parameters
    d_model = 64
    num_heads = 4  # Reduced from 8
    num_blocks = 2  # Reduced from 4
    d_ff = 64 * 2  # Reduced from 64*4
    
    # Calculate approximate parameter count
    # Simplified calculation for transformer
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
    
    print(f"📊 Model Complexity:")
    print(f"   Transformer blocks: {num_blocks}")
    print(f"   Attention heads: {num_heads}")
    print(f"   Model dimension: {d_model}")
    print(f"   Feedforward dim: {d_ff}")
    print(f"   Estimated parameters: ~{total_params:,}")
    
    # Dataset size analysis
    dataset_size = 20000  # New increased size
    train_size = int(dataset_size * 0.8)  # 80% train
    
    print(f"\n📈 Dataset Analysis:")
    print(f"   Total samples: {dataset_size:,}")
    print(f"   Training samples: {train_size:,}")
    print(f"   Samples per parameter: {train_size/total_params:.1f}")
    
    # Risk assessment
    if train_size / total_params > 10:
        risk = "LOW"
        color = "✅"
    elif train_size / total_params > 5:
        risk = "MEDIUM"
        color = "⚠️"
    else:
        risk = "HIGH"
        color = "❌"
    
    print(f"\n{color} Overfitting Risk: {risk}")
    
    if risk == "HIGH":
        print("   Recommendations:")
        print("   • Increase dataset size further")
        print("   • Reduce model complexity")
        print("   • Increase regularization")
    elif risk == "MEDIUM":
        print("   Recommendations:")
        print("   • Use early stopping")
        print("   • Monitor validation loss closely")
        print("   • Consider data augmentation")
    else:
        print("   Good balance between model and data size!")

def main():
    """Main function"""
    print("🧠 PatchedBrainTransformer - Overfitting Prevention Analysis")
    print("=" * 70)
    
    success = test_dataset_sizes()
    
    if success:
        analyze_overfitting_risk()
        
        print(f"\n🎯 Summary of Changes Made:")
        print("=" * 30)
        print("✅ Dataset size: 5,000 → 20,000 samples")
        print("✅ Test split: 5% → 20%")
        print("✅ Dropout: 0.1 → 0.2")
        print("✅ Weight decay: 0.01 → 0.05")
        print("✅ Transformer blocks: 4 → 2")
        print("✅ Attention heads: 8 → 4")
        print("✅ Added early stopping (patience=20)")
        print("✅ Enhanced data augmentation")
        
        print(f"\n🚀 Expected Results:")
        print("• Reduced overfitting")
        print("• Better generalization")
        print("• More stable training curves")
        print("• Automatic stopping when optimal")

if __name__ == "__main__":
    main()
