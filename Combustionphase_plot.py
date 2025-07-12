import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def phase_space_3d(data, sample_rate, start_time=1.5, duration=0.5, tau_ms=2):
    """
    最简单的3D相位图实现
    
    参数:
    data: 处理后的压力数据
    sample_rate: 采样率 (Hz)
    start_time: 开始时间 (秒)
    duration: 分析时长 (秒) 
    tau_ms: 时间延迟 (毫秒)
    """
    
    # 1. 选择数据段
    start_idx = int(start_time * sample_rate)
    end_idx = int((start_time + duration) * sample_rate)
    segment = data[start_idx:end_idx]
    
    # 2. 计算延迟样本数
    tau_samples = int(tau_ms * 1e-3 * sample_rate)
    
    # 3. 相空间重构 [x(t), x(t+τ), x(t+2τ)]
    N = len(segment) - 2 * tau_samples
    x = segment[0:N]
    y = segment[tau_samples:N+tau_samples]  
    z = segment[2*tau_samples:N+2*tau_samples]
    
    # 4. 3D绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹，每隔几个点取一个（减少绘图点数）
    skip = max(1, N//5000)  # 最多画5000个点
    ax.plot(x[::skip], y[::skip], z[::skip], 'b-', linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('x(t)')
    ax.set_ylabel(f'x(t+{tau_ms}ms)')  
    ax.set_zlabel(f'x(t+{2*tau_ms}ms)')
    ax.set_title(f'3D Phase Space ({start_time:.1f}s-{start_time+duration:.1f}s)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"分析了 {duration}s 数据，共 {N} 个相空间点")

# ==================== 使用示例 ====================
def main():
    # 假设你已经有了处理后的数据
    # processed_data = your_processed_pressure_data
    # sample_rate = your_sample_rate
    
    # 示例：读取你保存的数据
    processed_data = np.load('processed_data.npy')  # 如果你保存了的话
    
    # 或者直接从你的main函数中获取
    sample_rate = 1e5  # 1MHz，根据你的实际情况调整
    
    # 制作3D相位图
    phase_space_3d(processed_data, sample_rate, 
                   start_time=0.8,    # 从几秒开始
                   duration=0.3,      # 分析多少秒数据
                   tau_ms=0.2)          # 2毫秒延迟

if __name__ == "__main__":
    main()