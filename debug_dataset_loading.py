#!/usr/bin/env python3
"""
Debug dataset loading issues
"""

import sys
import os
import numpy as np
sys.path.append('src')

from utils import get_mindbigdata_eeg

def check_dataset_file():
    """Check if dataset file exists and is readable"""
    print("🔍 Dataset File Check")
    print("=" * 25)
    
    file_path = "datasets/EP1.01.txt"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        
        # Check if datasets directory exists
        if not os.path.exists("datasets"):
            print("❌ Directory 'datasets' does not exist")
            print("   Please create it and add the MindBigData file")
        else:
            print("✅ Directory 'datasets' exists")
            
            # List files in datasets directory
            files = os.listdir("datasets")
            print(f"   Files in datasets/: {files}")
            
            # Look for similar files
            ep_files = [f for f in files if f.startswith("EP")]
            if ep_files:
                print(f"   Found EP files: {ep_files}")
                print("   You may need to rename one of these to 'EP1.01.txt'")
        
        return False
    
    print(f"✅ File exists: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    if file_size == 0:
        print("❌ File is empty!")
        return False
    elif file_size < 1024:
        print("⚠️ File is very small, may be incomplete")
    
    # Check if file is readable
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            print(f"   First line: {first_line[:100]}...")
            
            # Count lines
            f.seek(0)
            line_count = sum(1 for _ in f)
            print(f"   Total lines: {line_count:,}")
            
    except Exception as e:
        print(f"❌ Cannot read file: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test dataset loading with debug info"""
    print(f"\n🧪 Dataset Loading Test")
    print("=" * 25)
    
    try:
        print("Attempting to load dataset...")
        result = get_mindbigdata_eeg(
            freq_min=0.5, freq_max=40, resample=250
        )
        
        if result is None:
            print("❌ Dataset loading returned None")
            return False
        
        data, labels, meta, channels = result
        
        print("✅ Dataset loaded successfully!")
        print(f"   Data shape: {data.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Meta shape: {meta.shape}")
        print(f"   Channels: {channels}")
        print(f"   Label distribution: {np.bincount(labels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_solutions():
    """Suggest solutions for common issues"""
    print(f"\n💡 Common Solutions")
    print("=" * 20)
    
    print("1. 📁 Dataset File Missing:")
    print("   • Download MindBigData MNIST dataset")
    print("   • Place file as: datasets/EP1.01.txt")
    print("   • Ensure file is not corrupted")
    
    print(f"\n2. 🔧 File Format Issues:")
    print("   • File should be tab-separated")
    print("   • Format: [id] [event] [device] [channel] [code] [size] [data]")
    print("   • Check first few lines for correct format")
    
    print(f"\n3. 📊 Data Quality Issues:")
    print("   • Ensure signals have enough samples (>100)")
    print("   • Check for valid MNIST digits (0-9)")
    print("   • Verify EEG data is numeric")
    
    print(f"\n4. 🚀 Alternative Solutions:")
    print("   • Use synthetic data for testing")
    print("   • Try different MindBigData file")
    print("   • Check file permissions")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print(f"\n🔧 Creating Sample Dataset")
    print("=" * 25)
    
    # Create datasets directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)
    
    sample_file = "datasets/EP1.01_sample.txt"
    
    # Create sample data in MindBigData format
    sample_data = [
        "# MindBigData MNIST Sample",
        "1001\t1001\tEP\tAF3\t5\t256\t" + ",".join([str(i*0.1) for i in range(256)]),
        "1002\t1001\tEP\tF7\t5\t256\t" + ",".join([str(i*0.1 + 1) for i in range(256)]),
        "1003\t1001\tEP\tF3\t5\t256\t" + ",".join([str(i*0.1 + 2) for i in range(256)]),
        "1004\t1002\tEP\tAF3\t7\t256\t" + ",".join([str(i*0.1 + 3) for i in range(256)]),
        "1005\t1002\tEP\tF7\t7\t256\t" + ",".join([str(i*0.1 + 4) for i in range(256)]),
        "1006\t1002\tEP\tF3\t7\t256\t" + ",".join([str(i*0.1 + 5) for i in range(256)]),
    ]
    
    try:
        with open(sample_file, 'w') as f:
            f.write('\n'.join(sample_data))
        
        print(f"✅ Created sample dataset: {sample_file}")
        print("   You can test with this file by renaming it to EP1.01.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample dataset: {e}")
        return False

def main():
    """Main debug function"""
    print("🧠 PatchedBrainTransformer - Dataset Loading Debug")
    print("=" * 60)
    
    # Check dataset file
    file_ok = check_dataset_file()
    
    if file_ok:
        # Test loading
        loading_ok = test_dataset_loading()
        
        if loading_ok:
            print(f"\n🎉 Dataset loading works correctly!")
        else:
            print(f"\n❌ Dataset loading failed")
            suggest_solutions()
    else:
        print(f"\n❌ Dataset file issues detected")
        suggest_solutions()
        
        # Offer to create sample dataset
        print(f"\n🔧 Would you like to create a sample dataset for testing?")
        create_sample_dataset()

if __name__ == "__main__":
    main()
