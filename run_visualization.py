#!/usr/bin/env python3
"""
Script untuk menjalankan visualisasi training PatchedBrainTransformer
Gunakan setelah training selesai untuk melihat hasil dan analisis
"""

import sys
import os
sys.path.append('plots')
sys.path.append('src')

from visualize_training import TrainingVisualizer

def main():
    """Main function untuk menjalankan visualisasi"""
    print("🧠 PatchedBrainTransformer - Training Visualization")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Opsi 1: Buat comprehensive report (jika ada terminal output)
    print("\n📊 Creating comprehensive training report...")
    try:
        visualizer.create_comprehensive_report()
        print("✅ Comprehensive report created successfully!")
    except Exception as e:
        print(f"❌ Error creating comprehensive report: {e}")
        print("Trying individual visualizations...")
    
    # Opsi 2: Plot training curves dari log file (jika ada)
    log_files = [
        "training.log",
        "output.log", 
        "terminal_output.txt"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n📈 Parsing training log from {log_file}...")
            try:
                with open(log_file, 'r') as f:
                    terminal_output = f.read()
                
                visualizer.parse_training_log(terminal_output)
                visualizer.plot_training_curves(f'plots/training_curves_from_{log_file.replace(".", "_")}.png')
                print(f"✅ Training curves created from {log_file}")
                break
            except Exception as e:
                print(f"❌ Error processing {log_file}: {e}")
                continue
    
    # Opsi 3: Load dan visualisasi model yang sudah disimpan
    model_paths = [
        "models/mnist_brain_transformer/final_199.pt",
        "models/mnist_brain_transformer/checkpoint_180.pt",
        "models/mnist_brain_transformer/checkpoint_160.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\n🔍 Analyzing saved model: {model_path}")
            try:
                # Load model dan buat evaluasi
                visualizer.evaluate_model(model_path, save_path='plots/model_evaluation/')
                print(f"✅ Model evaluation completed for {model_path}")
                break
            except Exception as e:
                print(f"❌ Error evaluating model {model_path}: {e}")
                continue
    
    # Opsi 4: Visualisasi data EEG (jika tersedia)
    try:
        print("\n🧠 Creating EEG data visualization...")
        from utils import get_mindbigdata_eeg
        
        data, labels, meta, channels = get_mindbigdata_eeg(
            freq_min=0.5, freq_max=40, resample=250
        )
        
        if data is not None:
            visualizer.visualize_eeg_data(data[:100], labels[:100], 'plots/eeg_visualization/')
            print("✅ EEG data visualization created")
        else:
            print("⚠️ No EEG data available for visualization")
            
    except Exception as e:
        print(f"❌ Error creating EEG visualization: {e}")
    
    print("\n🎉 Visualization script completed!")
    print("📁 Check the 'plots/' directory for generated visualizations")
    print("\nGenerated files may include:")
    print("  - training_curves.png")
    print("  - confusion_matrix.png") 
    print("  - accuracy_per_digit.png")
    print("  - eeg_samples_per_digit.png")
    print("  - average_eeg_per_digit.png")

if __name__ == "__main__":
    main()
