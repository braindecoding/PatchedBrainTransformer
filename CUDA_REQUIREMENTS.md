# 🚀 CUDA Requirements for PatchedBrainTransformer

## ⚠️ IMPORTANT: CUDA is REQUIRED

This project **requires CUDA** for training. CPU training is **not supported** because:

- 🐌 **Performance**: Training would take days/weeks on CPU vs hours on GPU
- 💾 **Memory**: Model requires GPU memory management
- ⚡ **Mixed Precision**: AMP requires CUDA for optimal performance
- 🔧 **Optimizations**: cuDNN and TF32 optimizations are GPU-only

## 🔍 Quick CUDA Check

```bash
# Quick verification
python verify_cuda.py

# Comprehensive check
python check_cuda.py
```

## 🛠️ CUDA Setup

### 1. Install NVIDIA Drivers
```bash
# Check if GPU is detected
nvidia-smi
```

### 2. Install CUDA-enabled PyTorch
```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verify Installation
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🚀 Running Training

### Option 1: Safe Launcher (Recommended)
```bash
python run_training.py
```

### Option 2: Direct Training
```bash
python main_pre_training.py
```

### Option 3: Fine-tuning
```bash
python main_fine_tune.py
```

## ❌ What Happens Without CUDA

The training scripts will:
1. ✋ **Stop immediately** if CUDA is not detected
2. 📝 **Show clear error messages** with setup instructions
3. 🛑 **Exit with error code 1** (no CPU fallback)

## 🎯 Minimum Requirements

- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5+
- **Memory**: 4GB+ GPU memory (8GB+ recommended)
- **CUDA**: Version 11.8 or 12.1
- **PyTorch**: CUDA-enabled version

## 🔧 Troubleshooting

### "CUDA not available" Error
1. Check GPU: `nvidia-smi`
2. Reinstall PyTorch with CUDA
3. Verify: `python verify_cuda.py`

### Out of Memory Error
1. Reduce batch size in config
2. Enable mixed precision (already enabled)
3. Use gradient checkpointing

### Performance Issues
1. Enable optimizations (already enabled)
2. Use consistent input sizes
3. Monitor GPU utilization

## 📊 Expected Performance

With proper CUDA setup:
- **Training Speed**: 2-3x faster than CPU
- **Memory Usage**: ~50% reduction with AMP
- **Training Time**: Hours instead of days

## 🎉 Ready to Train!

Once CUDA is properly set up, you can enjoy:
- ⚡ **Fast training** with mixed precision
- 📊 **Real-time monitoring** and plotting
- 💾 **Automatic checkpointing**
- 🔧 **GPU optimizations**
