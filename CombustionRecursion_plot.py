import numpy as np
import matplotlib.pyplot as plt

def recurrence_plot(data, sample_rate, start_time=1.5, duration=0.25, 
                   tau_ms=2, embed_dim=10, threshold_percent=10, max_points=2000):
    """
    递归图生成 
    """
    
    # 1. 选择数据段
    start_idx = int(start_time * sample_rate)
    end_idx = int((start_time + duration) * sample_rate)
    segment = data[start_idx:end_idx]
    print(f"选择数据段: {len(segment)} 个点 ({duration}s)")
    
    # 2. 计算延迟样本数
    tau_samples = int(tau_ms * 1e-3 * sample_rate)
    print(f"延迟样本数 τ = {tau_samples} 个点")
    
    # 3. 相空间重构
    N_raw = len(segment) - (embed_dim - 1) * tau_samples
    
    # 4. 控制计算规模
    if N_raw > max_points:
        step = N_raw // max_points
        indices = np.arange(0, N_raw, step)[:max_points]
        print(f"降采样: {N_raw} → {len(indices)} 个点")
    else:
        indices = np.arange(N_raw)
    
    N = len(indices)
    phase_space = np.zeros((N, embed_dim))
    
    for i in range(embed_dim):
        phase_space[:, i] = segment[indices + i*tau_samples]
    
    print(f"相空间矩阵: {N} × {embed_dim}")
    
    # 5. 计算距离矩阵
    print("计算距离矩阵...")
    distances = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            distances[i, j] = np.linalg.norm(phase_space[i] - phase_space[j])
        
        if (i + 1) % (N // 10) == 0:  # 显示进度
            print(f"进度: {(i+1)*100//N}%")
    
    # 6. 确定阈值和生成递归矩阵
    epsilon = np.percentile(distances.flatten(), threshold_percent)
    recurrence_matrix = (distances <= epsilon).astype(int)
    
    print(f"阈值 ε = {epsilon:.4f}")
    
    # 7. 绘制递归图
    plt.figure(figsize=(10, 10))
    plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
    plt.title(f'递归图\n时间: {start_time}s-{start_time+duration}s, '
              f'τ={tau_ms}ms, m={embed_dim}, ε={epsilon:.4f}')
    plt.xlabel('时间索引 i')
    plt.ylabel('时间索引 j')
    
    # 添加递归率信息
    RR = np.sum(recurrence_matrix) / (N*N) * 100
    plt.figtext(0.02, 0.02, f'递归率 RR = {RR:.2f}%', fontsize=12, 
                bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()
    
    print(f"递归率 RR: {RR:.2f}%")
    print("完成！")
    
    return recurrence_matrix

# ==================== 使用 ====================
def main():
    # 读取数据
    processed_data = np.load('processed_data.npy')
    sample_rate = 1e5  # 调整为你的实际采样率
    
    # 生成递归图
    rec_matrix = recurrence_plot(
        processed_data, 
        sample_rate,
        start_time=1.4,     # 开始时间
        duration=0.2,      # 分析时长
        tau_ms=2,          # 延迟时间
        embed_dim=10,      # 嵌入维数
        threshold_percent=10, # 阈值百分比
        max_points=2000    # 最大计算点数
    )

if __name__ == "__main__":
    main()