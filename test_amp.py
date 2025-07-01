#!/usr/bin/env python3
"""
Simple AMP test for PatchedBrainTransformer
"""

import torch

def test_amp_simple():
    """Simple AMP test"""
    print("🧪 Testing AMP for PatchedBrainTransformer")
    print("=" * 45)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test AMP")
        return False
    
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    
    # Try importing AMP
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("✅ AMP Import: SUCCESS (torch.cuda.amp)")
        
        # Test basic AMP functionality
        device = torch.device('cuda')
        
        # Create a simple model
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()
        
        # Test data
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 1, device=device)
        
        # Forward pass with autocast
        with autocast():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ AMP Test: SUCCESS")
        print(f"   Loss: {loss.item():.4f}")
        print("   🚀 AMP is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ AMP Test: FAILED - {e}")
        return False

def test_training_imports():
    """Test if training imports work"""
    print("\n🔍 Testing Training Imports")
    print("=" * 30)
    
    try:
        from src.train import training
        print("✅ Training import: SUCCESS")
        
        # Test if AMP variables are set correctly
        import src.train as train_module
        if hasattr(train_module, 'AMP_AVAILABLE'):
            print(f"✅ AMP_AVAILABLE: {train_module.AMP_AVAILABLE}")
        else:
            print("⚠️ AMP_AVAILABLE not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Training import: FAILED - {e}")
        return False

def main():
    """Main test function"""
    print("🧠 PatchedBrainTransformer - AMP Test")
    print("=" * 50)
    
    amp_works = test_amp_simple()
    imports_work = test_training_imports()
    
    print(f"\n📋 Test Summary:")
    print(f"   AMP Functionality: {'✅' if amp_works else '❌'}")
    print(f"   Training Imports: {'✅' if imports_work else '❌'}")
    
    if amp_works and imports_work:
        print("\n🎉 All tests passed! Ready for training with AMP.")
    elif imports_work:
        print("\n⚠️ Training will work but without AMP acceleration.")
    else:
        print("\n❌ Issues detected. Check PyTorch installation.")

if __name__ == "__main__":
    main()
