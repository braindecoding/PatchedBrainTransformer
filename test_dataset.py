#!/usr/bin/env python3
"""
Test script to check if MindBigData dataset can be loaded
"""

import sys
import os
sys.path.append('src')

from utils import get_mindbigdata_eeg

def test_mindbigdata():
    print("Testing MindBigData dataset loading...")
    
    try:
        data, labels, meta, channels = get_mindbigdata_eeg()
        
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Meta shape: {meta.shape}")
        print(f"Channels: {channels}")
        print(f"Unique labels: {set(labels)}")
        print(f"First few labels: {labels[:10]}")
        
        print("✓ MindBigData dataset loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading MindBigData dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_mindbigdata()
    if success:
        print("\nDataset test passed! You can now run the main training script.")
    else:
        print("\nDataset test failed. Please check the dataset file.")
