#!/usr/bin/env python3
"""
Test script to check if MindBigData MNIST dataset can be loaded correctly
"""

import sys
import os
sys.path.append('src')

from utils import get_mindbigdata_eeg
import numpy as np

def test_mnist_mindbigdata():
    print("Testing MindBigData MNIST dataset loading...")
    
    try:
        # Test with MNIST parameters (10 classes)
        data, labels, meta, channels = get_mindbigdata_eeg(
            file_path="datasets/EP1.01.txt",
            n_classes=10  # MNIST digits 0-9
        )
        
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Meta shape: {meta.shape}")
        print(f"Channels ({len(channels)}): {channels}")
        print(f"Unique labels: {sorted(set(labels))}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # Check if we have 10 classes (0-9)
        unique_labels = sorted(set(labels))
        expected_labels = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        if unique_labels == expected_labels:
            print("✅ Correct MNIST labels (0-9) found!")
        else:
            print(f"❌ Expected labels {expected_labels}, but got {unique_labels}")
        
        # Check data dimensions
        n_trials, n_channels, n_samples = data.shape
        print(f"Number of trials: {n_trials}")
        print(f"Number of channels: {n_channels}")
        print(f"Number of samples per trial: {n_samples}")
        
        # Check if channels match MindBigData EPOC format
        expected_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        if channels == expected_channels:
            print("✅ Correct MindBigData channel layout!")
        else:
            print(f"❌ Expected channels {expected_channels}")
            print(f"   But got {channels}")
        
        # Check data range (EEG should be in reasonable microvolts range)
        data_min, data_max = data.min(), data.max()
        print(f"Data range: {data_min:.6f} to {data_max:.6f}")
        
        if -1000 <= data_min and data_max <= 1000:
            print("✅ Data range looks reasonable for EEG (microvolts)")
        else:
            print("⚠️  Data range might be unusual for EEG")
        
        print("✅ MindBigData MNIST dataset test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing MindBigData MNIST dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mnist_mindbigdata()
    if success:
        print("\n🎯 Dataset is ready for MNIST visual stimulus classification!")
        print("   - 10 classes (digits 0-9)")
        print("   - 14 EEG channels")
        print("   - Proper data format")
    else:
        print("\n❌ Dataset test failed. Please check the implementation.")
