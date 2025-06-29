#!/usr/bin/env python3
"""
Visualisasi Training Progress dan Hasil Klasifikasi
PatchedBrainTransformer untuk MindBigData MNIST
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from utils import get_mindbigdata_eeg, SeqDataset
from model import PBT

# Set style untuk plot yang bagus
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    def __init__(self, log_file=None, model_path=None):
        """
        Initialize visualizer
        
        Args:
            log_file: Path ke file log training (opsional)
            model_path: Path ke model checkpoint (opsional)
        """
        self.log_file = log_file
        self.model_path = model_path
        self.training_history = []
        
    def parse_training_log(self, terminal_output=None):
        """Parse training log dari terminal output atau file"""
        if terminal_output:
            lines = terminal_output.split('\n')
        elif self.log_file and os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
        else:
            print("No training log available. Using dummy data for demonstration.")
            return self._create_dummy_training_data()
        
        epochs = []
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        
        for line in lines:
            line = line.strip()
            
            # Parse epoch dan loss
            if "Epoch:" in line and "train_loss" in line and "test_loss" in line:
                parts = line.split(',')
                try:
                    epoch = int(parts[0].split(':')[1].strip())
                    train_loss = float(parts[1].split('train_loss')[1].strip())
                    test_loss = float(parts[2].split('test_loss')[1].strip())
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                except:
                    continue
            
            # Parse accuracy
            elif "train_acc" in line and "test_acc" in line:
                parts = line.split(',')
                try:
                    train_acc = float(parts[0].split('train_acc')[1].strip()) * 100
                    test_acc = float(parts[1].split('test_acc')[1].strip()) * 100
                    
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                except:
                    continue
        
        # Ensure same length
        min_len = min(len(epochs), len(train_accs))
        self.training_history = {
            'epoch': epochs[:min_len],
            'train_loss': train_losses[:min_len],
            'test_loss': test_losses[:min_len],
            'train_acc': train_accs[:min_len],
            'test_acc': test_accs[:min_len]
        }
        
        return self.training_history
    
    def _create_dummy_training_data(self):
        """Create dummy training data for demonstration"""
        epochs = list(range(0, 50))
        train_losses = [2.3 - 0.01*i + 0.005*np.sin(i*0.5) + np.random.normal(0, 0.01) for i in epochs]
        test_losses = [2.3 - 0.008*i + 0.01*np.sin(i*0.3) + np.random.normal(0, 0.015) for i in epochs]
        train_accs = [10 + 2*i + 5*np.sin(i*0.2) + np.random.normal(0, 2) for i in epochs]
        test_accs = [10 + 1.5*i + 8*np.sin(i*0.15) + np.random.normal(0, 3) for i in epochs]
        
        # Clip accuracies to reasonable range
        train_accs = np.clip(train_accs, 5, 95)
        test_accs = np.clip(test_accs, 5, 95)
        
        self.training_history = {
            'epoch': epochs,
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_acc': train_accs,
            'test_acc': test_accs
        }
        
        return self.training_history
    
    def plot_training_curves(self, save_path='plots/training_curves.png'):
        """Plot training dan validation curves"""
        if not self.training_history:
            print("No training history available. Run parse_training_log() first.")
            return
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epoch']
        
        # Loss curves
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.training_history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.training_history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(self.training_history['test_loss']) - np.array(self.training_history['train_loss'])
        ax3.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Test Loss - Train Loss')
        ax3.set_title('Overfitting Indicator')
        ax3.grid(True, alpha=0.3)
        
        # Learning rate effect (smoothed loss)
        if len(epochs) > 5:
            smooth_train = pd.Series(self.training_history['train_loss']).rolling(window=5).mean()
            smooth_test = pd.Series(self.training_history['test_loss']).rolling(window=5).mean()
            ax4.plot(epochs, smooth_train, 'b-', label='Smoothed Train Loss', linewidth=2)
            ax4.plot(epochs, smooth_test, 'r-', label='Smoothed Test Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Smoothed Loss')
            ax4.set_title('Smoothed Loss Curves')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to: {save_path}")
    
    def load_model_and_data(self, config):
        """Load trained model and test data"""
        try:
            # Load test data
            print("Loading MindBigData test data...")
            test_data_set = SeqDataset(
                dim_token=config["d_input"],
                num_tokens_per_channel=config["num_tokens_per_channel"],
                reduce_num_chs_to=30,
                augmentation=[],  # No augmentation for testing
            )
            
            # Add test data
            data_array, labels, meta_df, channels = get_mindbigdata_eeg(
                file_path="datasets/EP1.01.txt",
                resample=250,
                channels=None,
                n_channels=14
            )
            
            # Use last 100 samples for testing
            test_data_set.append_data_set(
                data_array[-100:], channels, labels[-100:]
            )
            
            # Load model if checkpoint exists
            model = None
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading model from: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                model = PBT(
                    d_input=config["d_input"],
                    n_classes=10,
                    num_embeddings=64,
                    num_tokens_per_channel=config["num_tokens_per_channel"],
                    d_model=config["d_model"],
                    n_blocks=config["num_transformer_blocks"],
                    num_heads=config["num_heads"],
                    bias_transformer=config["bias_transformer"],
                    dropout=config["dropout"],
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print("Model loaded successfully!")
            
            return model, test_data_set, data_array[-100:], labels[-100:]
            
        except Exception as e:
            print(f"Error loading model/data: {e}")
            return None, None, None, None

    def visualize_predictions(self, model, test_dataset, test_data, test_labels, save_path='plots/'):
        """Visualize model predictions and confusion matrix"""
        if model is None:
            print("No model available for prediction visualization.")
            return

        os.makedirs(save_path, exist_ok=True)

        # Get predictions
        print("Getting model predictions...")
        predictions = []
        true_labels = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        with torch.no_grad():
            for i in range(min(len(test_dataset), 100)):  # Limit to 100 samples
                try:
                    data, label, pos = test_dataset[i]
                    data = data.unsqueeze(0).to(device)
                    pos = pos.unsqueeze(0).to(device)

                    output = model(data, pos)
                    pred = torch.argmax(output, dim=1).cpu().item()

                    predictions.append(pred)
                    true_labels.append(label.item())
                except:
                    continue

        if len(predictions) == 0:
            print("No predictions could be made.")
            return

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix - MNIST Digit Classification from EEG')
        plt.xlabel('Predicted Digit')
        plt.ylabel('True Digit')
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions,
                                  target_names=[f'Digit {i}' for i in range(10)]))

        # Accuracy per digit
        plt.figure(figsize=(12, 6))
        digit_accuracy = []
        for digit in range(10):
            digit_mask = np.array(true_labels) == digit
            if np.sum(digit_mask) > 0:
                digit_acc = np.mean(np.array(predictions)[digit_mask] == digit) * 100
                digit_accuracy.append(digit_acc)
            else:
                digit_accuracy.append(0)

        bars = plt.bar(range(10), digit_accuracy, color=sns.color_palette("husl", 10))
        plt.xlabel('MNIST Digit')
        plt.ylabel('Accuracy (%)')
        plt.title('Classification Accuracy per MNIST Digit')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.savefig(f'{save_path}/accuracy_per_digit.png', dpi=300, bbox_inches='tight')
        plt.show()

        return predictions, true_labels

    def visualize_eeg_data(self, eeg_data, labels, save_path='plots/'):
        """Visualize raw EEG data samples"""
        os.makedirs(save_path, exist_ok=True)

        # Channel names for EPOC
        channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

        # Plot sample EEG signals for each digit
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            if len(digit_indices) > 0:
                # Take first sample of this digit
                sample_idx = digit_indices[0]
                sample_data = eeg_data[sample_idx]  # Shape: (14, 256)

                # Plot all channels
                for ch_idx in range(min(14, len(channels))):
                    axes[digit].plot(sample_data[ch_idx], alpha=0.7, linewidth=0.8)

                axes[digit].set_title(f'Digit {digit} - EEG Signals')
                axes[digit].set_xlabel('Time (samples)')
                axes[digit].set_ylabel('Amplitude')
                axes[digit].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/eeg_samples_per_digit.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot average EEG response per digit
        plt.figure(figsize=(15, 10))

        for digit in range(10):
            digit_indices = np.where(labels == digit)[0]
            if len(digit_indices) > 0:
                # Average across all samples of this digit
                avg_response = np.mean(eeg_data[digit_indices], axis=0)  # Shape: (14, 256)

                # Plot average response for each channel
                plt.subplot(2, 5, digit + 1)
                for ch_idx in range(min(14, len(channels))):
                    plt.plot(avg_response[ch_idx], alpha=0.8, linewidth=1,
                           label=channels[ch_idx] if digit == 0 else "")

                plt.title(f'Digit {digit} - Average EEG Response')
                plt.xlabel('Time (samples)')
                plt.ylabel('Amplitude')
                plt.grid(True, alpha=0.3)

                if digit == 0:  # Add legend only to first subplot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{save_path}/average_eeg_per_digit.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_comprehensive_report(self, terminal_output=None):
        """Create comprehensive training and evaluation report"""
        print("🧠 PatchedBrainTransformer - MindBigData MNIST Analysis Report")
        print("=" * 70)

        # Parse training history
        print("\n📈 Parsing training history...")
        self.parse_training_log(terminal_output)

        # Plot training curves
        print("\n📊 Creating training curves...")
        self.plot_training_curves()

        # Load model and data for evaluation
        config = {
            "d_input": 32,
            "d_model": 64,
            "num_tokens_per_channel": 7,
            "num_transformer_blocks": 4,
            "num_heads": 8,
            "bias_transformer": True,
            "dropout": 0.1,
        }

        print("\n🔄 Loading model and test data...")
        model, test_dataset, test_data, test_labels = self.load_model_and_data(config)

        if model is not None and test_dataset is not None:
            print("\n🎯 Generating prediction visualizations...")
            self.visualize_predictions(model, test_dataset, test_data, test_labels)

        if test_data is not None and test_labels is not None:
            print("\n🧠 Visualizing EEG data patterns...")
            self.visualize_eeg_data(test_data, test_labels)

        print("\n✅ Comprehensive analysis complete!")
        print("📁 All plots saved in 'plots/' directory")


def main():
    """Main function to run visualization"""
    # Initialize visualizer
    visualizer = TrainingVisualizer()

    # Example terminal output (you can replace this with actual training log)
    sample_terminal_output = """
    Epoch: 0, train_loss 2.302784097605738, test_loss 2.3009417057037354
    Epoch: 0, train_acc 0.09482758492231369, test_acc 0.1599999964237213
    Epoch: 1, train_loss 2.303178984543373, test_loss 2.3008904457092285
    Epoch: 1, train_acc 0.08512931317090988, test_acc 0.11999999731779099
    Epoch: 2, train_loss 2.3027014896787446, test_loss 2.300847053527832
    Epoch: 2, train_acc 0.10344827175140381, test_acc 0.17999999225139618
    """

    # Create comprehensive report
    visualizer.create_comprehensive_report(sample_terminal_output)


if __name__ == "__main__":
    main()
