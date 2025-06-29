#!/usr/bin/env python3
"""
Visualisasi Pola EEG untuk MindBigData MNIST
Menganalisis pola sinyal otak untuk setiap digit MNIST
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from utils import get_mindbigdata_eeg

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_eeg_data():
    """Load MindBigData EEG data"""
    print("🧠 Loading MindBigData EPOC data...")
    
    try:
        data_array, labels, meta_df, channels = get_mindbigdata_eeg(
            file_path="datasets/EP1.01.txt",
            resample=250,
            channels=None
        )
        
        print(f"✅ Loaded {len(data_array)} trials")
        print(f"📊 Data shape: {data_array[0].shape if len(data_array) > 0 else 'No data'}")
        print(f"🔢 Digit distribution: {np.bincount(labels)}")
        
        return data_array, labels, channels
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def analyze_eeg_patterns(data_array, labels, channels):
    """Analyze EEG patterns for each digit"""
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Convert to numpy array for easier processing
    if isinstance(data_array, list):
        # Find minimum length to ensure consistent shape
        min_length = min(trial.shape[1] for trial in data_array)
        eeg_data = np.array([trial[:, :min_length] for trial in data_array])
    else:
        eeg_data = data_array
    
    labels = np.array(labels)
    
    print(f"📊 EEG data shape: {eeg_data.shape}")
    print(f"🔢 Labels shape: {labels.shape}")
    
    # 1. Average EEG Response per Digit
    plt.figure(figsize=(20, 12))
    
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) == 0:
            continue
            
        # Average across all trials for this digit
        avg_response = np.mean(eeg_data[digit_indices], axis=0)  # Shape: (14, time_samples)
        
        plt.subplot(2, 5, digit + 1)
        
        # Plot each channel
        for ch_idx, channel in enumerate(channels[:14]):
            plt.plot(avg_response[ch_idx], alpha=0.7, linewidth=1, label=channel if digit == 0 else "")
        
        plt.title(f'Digit {digit} - Average EEG Response\n({len(digit_indices)} trials)')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude (µV)')
        plt.grid(True, alpha=0.3)
        
        if digit == 0:  # Add legend only to first subplot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots/average_eeg_per_digit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Channel-wise Analysis
    plt.figure(figsize=(16, 10))
    
    # Calculate average power for each channel and digit
    channel_power = np.zeros((10, 14))  # 10 digits, 14 channels
    
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) == 0:
            continue
            
        for ch_idx in range(14):
            # Calculate RMS power for this channel and digit
            channel_data = eeg_data[digit_indices, ch_idx, :]
            channel_power[digit, ch_idx] = np.sqrt(np.mean(channel_data**2))
    
    # Heatmap of channel power per digit
    plt.subplot(2, 2, 1)
    sns.heatmap(channel_power, 
                xticklabels=channels[:14], 
                yticklabels=[f'Digit {i}' for i in range(10)],
                cmap='viridis', annot=False, cbar_kws={'label': 'RMS Power (µV)'})
    plt.title('Channel Power per Digit')
    plt.xlabel('EEG Channel')
    plt.ylabel('MNIST Digit')
    
    # 3. Frequency Analysis
    plt.subplot(2, 2, 2)
    
    # Calculate power spectral density for each digit
    freqs = np.fft.fftfreq(eeg_data.shape[2], 1/250)[:eeg_data.shape[2]//2]  # Assuming 250Hz sampling
    
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) == 0:
            continue
        
        # Average across all channels and trials for this digit
        digit_data = np.mean(eeg_data[digit_indices], axis=(0, 1))  # Average across trials and channels
        
        # Compute power spectral density
        psd = np.abs(np.fft.fft(digit_data)[:len(freqs)])**2
        
        plt.plot(freqs, psd, alpha=0.7, label=f'Digit {digit}')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Frequency Content per Digit')
    plt.xlim(0, 50)  # Focus on 0-50 Hz
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Temporal Dynamics
    plt.subplot(2, 2, 3)
    
    # Calculate temporal variance for each digit
    time_variance = np.zeros((10, eeg_data.shape[2]))
    
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) == 0:
            continue
        
        # Variance across trials and channels at each time point
        digit_data = eeg_data[digit_indices]  # Shape: (trials, channels, time)
        time_variance[digit] = np.var(digit_data, axis=(0, 1))  # Variance across trials and channels
    
    for digit in range(10):
        if np.any(time_variance[digit] > 0):
            plt.plot(time_variance[digit], alpha=0.7, label=f'Digit {digit}')
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Variance across trials/channels')
    plt.title('Temporal Dynamics per Digit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Channel Correlation Analysis
    plt.subplot(2, 2, 4)
    
    # Calculate correlation between channels for each digit
    correlations = []
    
    for digit in range(10):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) == 0:
            continue
        
        # Average across trials for this digit
        digit_avg = np.mean(eeg_data[digit_indices], axis=0)  # Shape: (channels, time)
        
        # Calculate correlation matrix between channels
        corr_matrix = np.corrcoef(digit_avg)
        
        # Take upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        correlations.append(np.mean(upper_tri))
    
    plt.bar(range(10), correlations, alpha=0.7, color=sns.color_palette("husl", 10))
    plt.xlabel('MNIST Digit')
    plt.ylabel('Average Inter-channel Correlation')
    plt.title('Channel Synchronization per Digit')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/eeg_channel_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return channel_power, correlations

def create_brain_topography(channel_power, channels):
    """Create brain topography visualization"""
    
    # EPOC channel positions (approximate 2D coordinates)
    channel_positions = {
        'AF3': (-0.3, 0.8), 'AF4': (0.3, 0.8),
        'F7': (-0.7, 0.4), 'F3': (-0.3, 0.4), 'F4': (0.3, 0.4), 'F8': (0.7, 0.4),
        'FC5': (-0.5, 0.2), 'FC6': (0.5, 0.2),
        'T7': (-0.8, 0), 'T8': (0.8, 0),
        'P7': (-0.7, -0.4), 'P8': (0.7, -0.4),
        'O1': (-0.3, -0.8), 'O2': (0.3, -0.8)
    }
    
    plt.figure(figsize=(20, 8))
    
    for digit in range(10):
        plt.subplot(2, 5, digit + 1)
        
        # Create head outline
        head_circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        plt.gca().add_patch(head_circle)
        
        # Plot channels
        for ch_idx, channel in enumerate(channels[:14]):
            if channel in channel_positions:
                x, y = channel_positions[channel]
                power = channel_power[digit, ch_idx]
                
                # Normalize power for color mapping
                norm_power = (power - np.min(channel_power)) / (np.max(channel_power) - np.min(channel_power))
                
                plt.scatter(x, y, s=200, c=norm_power, cmap='viridis', 
                           vmin=0, vmax=1, alpha=0.8, edgecolors='black')
                plt.text(x, y-0.15, channel, ha='center', va='center', fontsize=8)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'Digit {digit} - Brain Activity')
    
    # Add colorbar
    plt.tight_layout()
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gcf().get_axes(), shrink=0.8)
    cbar.set_label('Normalized EEG Power', rotation=270, labelpad=20)
    
    plt.savefig('plots/brain_topography.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("🧠 EEG Pattern Analysis for MindBigData MNIST")
    print("=" * 60)
    
    # Load data
    data_array, labels, channels = load_eeg_data()
    
    if data_array is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Analyze patterns
    print("\n📊 Analyzing EEG patterns...")
    channel_power, correlations = analyze_eeg_patterns(data_array, labels, channels)
    
    # Create brain topography
    print("\n🧠 Creating brain topography...")
    create_brain_topography(channel_power, channels)
    
    # Summary
    print("\n📋 ANALYSIS SUMMARY:")
    print("=" * 40)
    print(f"📊 Analyzed {len(data_array)} EEG trials")
    print(f"🔢 10 MNIST digits (0-9)")
    print(f"🧠 14 EPOC channels")
    print(f"⏱️  {data_array[0].shape[1] if len(data_array) > 0 else 'N/A'} time samples per trial")
    
    # Find most/least active channels
    avg_power = np.mean(channel_power, axis=0)
    most_active = channels[np.argmax(avg_power)]
    least_active = channels[np.argmin(avg_power)]
    
    print(f"🔥 Most active channel: {most_active}")
    print(f"❄️  Least active channel: {least_active}")
    
    # Find most/least synchronized digits
    most_sync_digit = np.argmax(correlations)
    least_sync_digit = np.argmin(correlations)
    
    print(f"🔗 Most synchronized digit: {most_sync_digit}")
    print(f"🔀 Least synchronized digit: {least_sync_digit}")
    
    print("\n✅ Analysis complete!")
    print("📁 Visualizations saved in 'plots/' directory:")
    print("   • average_eeg_per_digit.png")
    print("   • eeg_channel_analysis.png") 
    print("   • brain_topography.png")

if __name__ == "__main__":
    main()
