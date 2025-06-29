#!/usr/bin/env python3
"""
Real-time Training Monitor untuk PatchedBrainTransformer
Mengambil data dari terminal yang sedang berjalan dan membuat visualisasi
"""

import sys
import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RealTimeTrainingMonitor:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        
        # Setup plot
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🧠 PatchedBrainTransformer - Real-time Training Monitor', fontsize=16)
        
    def parse_terminal_output(self, output_text):
        """Parse training metrics from terminal output"""
        lines = output_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse epoch dan loss
            if "Epoch:" in line and "train_loss" in line and "test_loss" in line:
                try:
                    # Extract epoch
                    epoch_match = re.search(r'Epoch:\s*(\d+)', line)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                    
                    # Extract train_loss
                    train_loss_match = re.search(r'train_loss\s+([\d.]+)', line)
                    if train_loss_match:
                        train_loss = float(train_loss_match.group(1))
                    
                    # Extract test_loss
                    test_loss_match = re.search(r'test_loss\s+([\d.]+)', line)
                    if test_loss_match:
                        test_loss = float(test_loss_match.group(1))
                    
                    # Only add if we have all values and it's a new epoch
                    if epoch not in self.epochs:
                        self.epochs.append(epoch)
                        self.train_losses.append(train_loss)
                        self.test_losses.append(test_loss)
                        
                except Exception as e:
                    continue
            
            # Parse accuracy
            elif "train_acc" in line and "test_acc" in line:
                try:
                    # Extract train_acc
                    train_acc_match = re.search(r'train_acc\s+([\d.]+)', line)
                    if train_acc_match:
                        train_acc = float(train_acc_match.group(1)) * 100
                    
                    # Extract test_acc
                    test_acc_match = re.search(r'test_acc\s+([\d.]+)', line)
                    if test_acc_match:
                        test_acc = float(test_acc_match.group(1)) * 100
                    
                    # Add accuracy for the last epoch
                    if len(self.train_accs) < len(self.epochs):
                        self.train_accs.append(train_acc)
                        self.test_accs.append(test_acc)
                        
                except Exception as e:
                    continue
    
    def update_plots(self):
        """Update all plots with current data"""
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        if len(self.epochs) == 0:
            return
        
        # Ensure all arrays have same length
        min_len = min(len(self.epochs), len(self.train_losses), len(self.test_losses), 
                     len(self.train_accs), len(self.test_accs))
        
        epochs = self.epochs[:min_len]
        train_losses = self.train_losses[:min_len]
        test_losses = self.test_losses[:min_len]
        train_accs = self.train_accs[:min_len]
        test_accs = self.test_accs[:min_len]
        
        # Plot 1: Loss curves
        self.ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        self.ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, marker='s')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training & Test Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Add current values as text
        if len(train_losses) > 0:
            self.ax1.text(0.02, 0.98, f'Current Train Loss: {train_losses[-1]:.4f}', 
                         transform=self.ax1.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            self.ax1.text(0.02, 0.88, f'Current Test Loss: {test_losses[-1]:.4f}', 
                         transform=self.ax1.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Plot 2: Accuracy curves
        self.ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o')
        self.ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2, marker='s')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.set_title('Training & Test Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Add current values as text
        if len(train_accs) > 0:
            self.ax2.text(0.02, 0.98, f'Current Train Acc: {train_accs[-1]:.2f}%', 
                         transform=self.ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            self.ax2.text(0.02, 0.88, f'Current Test Acc: {test_accs[-1]:.2f}%', 
                         transform=self.ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Plot 3: Loss difference (overfitting indicator)
        if len(train_losses) > 0:
            loss_diff = np.array(test_losses) - np.array(train_losses)
            self.ax3.plot(epochs, loss_diff, 'g-', linewidth=2, marker='d')
            self.ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax3.set_xlabel('Epoch')
            self.ax3.set_ylabel('Test Loss - Train Loss')
            self.ax3.set_title('Overfitting Indicator')
            self.ax3.grid(True, alpha=0.3)
            
            # Color coding for overfitting
            current_diff = loss_diff[-1] if len(loss_diff) > 0 else 0
            color = 'red' if current_diff > 0.1 else 'orange' if current_diff > 0.05 else 'green'
            status = 'Overfitting' if current_diff > 0.1 else 'Slight Overfitting' if current_diff > 0.05 else 'Good'
            
            self.ax3.text(0.02, 0.98, f'Status: {status}', 
                         transform=self.ax3.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        # Plot 4: Learning progress summary
        if len(epochs) > 1:
            # Calculate improvement rates
            train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            test_improvement = (test_losses[0] - test_losses[-1]) / test_losses[0] * 100
            acc_improvement = test_accs[-1] - test_accs[0] if len(test_accs) > 1 else 0
            
            # Bar plot of improvements
            categories = ['Train Loss\nImprovement', 'Test Loss\nImprovement', 'Test Accuracy\nGain']
            values = [train_improvement, test_improvement, acc_improvement]
            colors = ['blue', 'red', 'green']
            
            bars = self.ax4.bar(categories, values, color=colors, alpha=0.7)
            self.ax4.set_ylabel('Improvement (%)')
            self.ax4.set_title('Learning Progress Summary')
            self.ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.draw()
    
    def save_current_state(self, save_path='plots/current_training_state.png'):
        """Save current training state"""
        os.makedirs('plots', exist_ok=True)
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training state saved to: {save_path}")
    
    def print_summary(self):
        """Print current training summary"""
        if len(self.epochs) == 0:
            print("No training data available yet.")
            return
        
        print("\n" + "="*60)
        print("🧠 MINDBIDATA MNIST BRAIN TRANSFORMER - TRAINING SUMMARY")
        print("="*60)
        print(f"📊 Current Epoch: {self.epochs[-1]}")
        print(f"📈 Train Loss: {self.train_losses[-1]:.4f}")
        print(f"📉 Test Loss: {self.test_losses[-1]:.4f}")
        print(f"🎯 Train Accuracy: {self.train_accs[-1]:.2f}%")
        print(f"🎯 Test Accuracy: {self.test_accs[-1]:.2f}%")
        
        if len(self.epochs) > 1:
            loss_improvement = (self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100
            acc_improvement = self.test_accs[-1] - self.test_accs[0]
            print(f"📊 Loss Improvement: {loss_improvement:.2f}%")
            print(f"📈 Accuracy Gain: +{acc_improvement:.2f}%")
        
        print("="*60)


def monitor_from_sample_data():
    """Demo function with sample training data"""
    monitor = RealTimeTrainingMonitor()
    
    # Sample terminal output (replace with actual training log)
    sample_output = """
    Epoch: 0, train_loss 2.302784097605738, test_loss 2.3009417057037354
    Epoch: 0, train_acc 0.09482758492231369, test_acc 0.1599999964237213
    Epoch: 1, train_loss 2.303178984543373, test_loss 2.3008904457092285
    Epoch: 1, train_acc 0.08512931317090988, test_acc 0.11999999731779099
    Epoch: 2, train_loss 2.3027014896787446, test_loss 2.300847053527832
    Epoch: 2, train_acc 0.10344827175140381, test_acc 0.17999999225139618
    Epoch: 3, train_loss 2.3027498968716325, test_loss 2.3007140159606934
    Epoch: 3, train_acc 0.09482758492231369, test_acc 0.14000000059604645
    """
    
    # Parse the sample data
    monitor.parse_terminal_output(sample_output)
    
    # Update plots
    monitor.update_plots()
    
    # Print summary
    monitor.print_summary()
    
    # Save current state
    monitor.save_current_state()
    
    # Show plots
    plt.show()
    
    return monitor


if __name__ == "__main__":
    print("🧠 Starting Real-time Training Monitor...")
    monitor = monitor_from_sample_data()
