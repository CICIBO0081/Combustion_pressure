#!/usr/bin/env python3
"""
Simple Pressure Sensor Data Processor
- Read 1MHz data (entire file)
- Convert to pressure: raw * 1.42 * 1e6
- Bandpass filter: 100-4000Hz
- Downsample to 100kHz (optional - comment out to disable)
- Normalize to (-1, 1) range
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==================== 可视化参数 ====================
# 1. 时域完整图显示区间（秒）
TIME_FULL_START = 0.0      
TIME_FULL_END = 7.0        

# 2. 时域细节图显示区间（秒）
TIME_DETAIL_START = 1.2
TIME_DETAIL_END = 2.2  

# 3. 频域分析数据区间（秒）- 用这段数据计算FFT
FFT_DATA_START = 1     
FFT_DATA_END = 3        

# 4. 频域显示范围（Hz）
FREQ_MIN = 0              
FREQ_MAX = 1200          

# 5. 频谱显示方式
USE_SEMILOGY = False      # True: 对数坐标, False: 线性坐标

# ==================== Core Functions ====================
def read_data(filename):
    """Read all sensor data from file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(float(line.strip()))
            except:
                continue
    return np.array(data)

def process_data(raw_data, sample_rate=1e6):
    """Process the data"""
    print("Processing steps:")
    
    # Step 1: Convert to pressure
    pressure = raw_data * 1.42 * 1e6
    print(f"1. Converted to pressure (Pa)")
    
    # Step 2: Remove DC offset
    pressure = pressure - np.mean(pressure)
    print(f"2. Removed DC offset")
    
    # Step 3: Bandpass filter (100-4000 Hz)
    nyquist = sample_rate / 2
    low_freq = 200 / nyquist
    high_freq = 4000 / nyquist
    
    sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
    pressure_filtered = signal.sosfiltfilt(sos, pressure)
    print(f"3. Applied bandpass filter (100-4000 Hz)")
    
    # # ========== DOWNSAMPLE SECTION - COMMENT OUT TO DISABLE ==========
    # Step 4: Downsample by 10
    sos_anti = signal.butter(8, 0.09, btype='low', output='sos')
    pressure_anti = signal.sosfiltfilt(sos_anti, pressure_filtered)
    
    pressure_downsampled = pressure_anti[::10]
    new_sample_rate = sample_rate / 10
    print(f"4. Downsampled from {int(sample_rate/1000)}kHz to {int(new_sample_rate/1000)}kHz")
    
    # Step 5: Normalize to (-1, 1) range
    p_max = np.max(np.abs(pressure_downsampled))
    pressure_normalized = pressure_downsampled / p_max
    print(f"5. Normalized to (-1, 1) range (p_max = {p_max:.2e} Pa)")
    
    # Return normalized data
    return pressure_normalized, new_sample_rate, p_max
    # ========== END OF DOWNSAMPLE SECTION ==========
    
    # # If downsampling is commented out, uncomment these lines:
    # print(f"4. No downsampling - keeping original {int(sample_rate/1000)}kHz")
    # # Step 5: Normalize to (-1, 1) range
    # p_max = np.max(np.abs(pressure_filtered))
    # pressure_normalized = pressure_filtered / p_max
    # print(f"5. Normalized to (-1, 1) range (p_max = {p_max:.2e} Pa)")
    # return pressure_normalized, sample_rate, p_max

def plot_results(data, sample_rate, p_max):
    """Create plots with adjustable time windows"""
    time = np.arange(len(data)) / sample_rate
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Full signal (with adjustable window)
    plt.subplot(3, 1, 1)
    idx_start = int(TIME_FULL_START * sample_rate)
    idx_end = min(int(TIME_FULL_END * sample_rate), len(data))
    plt.plot(time[idx_start:idx_end], data[idx_start:idx_end], 'b-', linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Pressure (p/p_max)')
    plt.title(f'Combustion Chamber Pressure (Normalized) - Full Signal ({TIME_FULL_START:.2f}s to {TIME_FULL_END:.2f}s)')
    plt.grid(True, alpha=0.3)
    plt.xlim(TIME_FULL_START, TIME_FULL_END)
    plt.ylim(-1.1, 1.1)
    
    # Plot 2: Detail view (with adjustable window)
    plt.subplot(3, 1, 2)
    idx_detail_start = int(TIME_DETAIL_START * sample_rate)
    idx_detail_end = min(int(TIME_DETAIL_END * sample_rate), len(data))
    plt.plot(time[idx_detail_start:idx_detail_end], data[idx_detail_start:idx_detail_end], 'b-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Pressure (p/p_max)')
    plt.title(f'Zoomed View ({TIME_DETAIL_START:.3f}s to {TIME_DETAIL_END:.3f}s)')
    plt.grid(True, alpha=0.3)
    plt.xlim(TIME_DETAIL_START, TIME_DETAIL_END)
    plt.ylim(-1.1, 1.1)
    
    # Plot 3: Frequency spectrum (from selected data segment)
    plt.subplot(3, 1, 3)
    idx_fft_start = int(FFT_DATA_START * sample_rate)
    idx_fft_end = min(int(FFT_DATA_END * sample_rate), len(data))
    fft_data = data[idx_fft_start:idx_fft_end]
    
    n = len(fft_data)
    freqs = np.fft.fftfreq(n, 1/sample_rate)[:n//2]
    fft = np.abs(np.fft.fft(fft_data))[:n//2]
    
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    if USE_SEMILOGY:
        plt.semilogy(freqs[mask], fft[mask], 'r-', linewidth=1)
    else:
        plt.plot(freqs[mask], fft[mask], 'r-', linewidth=1)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum (from {FFT_DATA_START:.2f}s to {FFT_DATA_END:.2f}s)')
    plt.grid(True, alpha=0.3)
    plt.xlim(FREQ_MIN, FREQ_MAX)
    
    plt.axvline(x=100, color='g', linestyle='--', alpha=0.5, label='Filter bounds')
    plt.axvline(x=4000, color='g', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pressure_analysis.png', dpi=150)
    plt.show()
    
    # Print statistics
    print("\nData Statistics:")
    print(f"  Duration: {time[-1]:.2f} seconds")
    print(f"  Sample rate: {sample_rate/1000:.0f} kHz")
    print(f"  Normalization factor (p_max): {p_max:.2e} Pa")
    print(f"  Mean: {np.mean(data):.4f} (normalized)")
    print(f"  Std:  {np.std(data):.4f} (normalized)")
    print(f"  Max:  {np.max(data):.4f} (normalized)")
    print(f"  Min:  {np.min(data):.4f} (normalized)")

def save_data(data, sample_rate, p_max):
    """Save processed data"""
    np.save('processed_data.npy', data)
    
    with open('processed_info.txt', 'w') as f:
        f.write(f'Sample Rate: {sample_rate} Hz\n')
        f.write(f'Duration: {len(data)/sample_rate:.3f} seconds\n')
        f.write(f'Normalization factor (p_max): {p_max:.6e} Pa\n')
        f.write(f'Data is normalized to (-1, 1) range\n')
        f.write(f'To convert back to Pa: multiply by {p_max:.6e}\n')
    
    print("\nSaved: processed_data.npy and processed_info.txt")

# ==================== Main ====================
def main():
    input_file = 'PCB_H2\采集数据05-19-12时间 0730.txt'  # Change to your file name
    sample_rate = 1e6  # 1 MHz
    
    print("=== Pressure Sensor Data Processor (with Normalization) ===\n")
    
    # Read entire file
    print(f"Reading {input_file}...")
    raw_data = read_data(input_file)
    print(f"Read {len(raw_data)} samples ({len(raw_data)/sample_rate:.2f} seconds)")
    
    # Process
    print("\nProcessing...")
    processed_data, new_rate, p_max = process_data(raw_data, sample_rate)
    print(f"Processed {len(processed_data)} samples")
    
    # Save
    save_data(processed_data, new_rate, p_max)
    
    # Plot
    print("\nCreating plots...")
    plot_results(processed_data, new_rate, p_max)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
