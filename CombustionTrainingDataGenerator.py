import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle

class TrainingDataGenerator:
    """从典型样本生成训练数据集"""
    
    def __init__(self, sample_rate=1e5):
        self.sample_rate = sample_rate
        self.base_dir = "typical_samples"
        self.mode_names = {
            0: "mode0_stable",
            1: "mode1_intermittent", 
            2: "mode2_limit_cycle",
            3: "mode3_beat"
        }
        
        # 存储所有数据
        self.X = []  # 特征数据
        self.y = []  # 标签
        self.scaler = StandardScaler()
        
    def sliding_window(self, data, window_size, step_size):
        """滑动窗口切分数据"""
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data[i:i + window_size]
            windows.append(window)
            
        return np.array(windows)
    
    def load_and_process_samples(self, window_size=2000, step_size=200):
        """
        加载并处理所有样本
        
        参数:
        - window_size: 窗口大小（数据点数）
        - step_size: 步长（数据点数）
        """
        print(f"开始生成训练数据...")
        print(f"窗口大小: {window_size} 点 ({window_size/self.sample_rate:.3f}s)")
        print(f"步长: {step_size} 点 ({step_size/self.sample_rate:.3f}s)")
        print("-" * 50)
        
        total_windows = 0
        
        for mode, mode_name in self.mode_names.items():
            mode_dir = os.path.join(self.base_dir, mode_name)
            
            if not os.path.exists(mode_dir):
                print(f"⚠️  模式 {mode} 文件夹不存在: {mode_dir}")
                continue
            
            # 获取该模式下的所有.npy文件
            npy_files = [f for f in os.listdir(mode_dir) if f.endswith('.npy')]
            
            mode_windows = 0
            print(f"处理模式 {mode} ({mode_name}):")
            
            for npy_file in npy_files:
                data_path = os.path.join(mode_dir, npy_file)
                
                # 加载数据
                data = np.load(data_path)
                
                # 检查数据长度
                if len(data) < window_size:
                    print(f"  ⚠️  跳过 {npy_file}: 数据长度({len(data)}) < 窗口大小({window_size})")
                    continue
                
                # 滑动窗口切分
                windows = self.sliding_window(data, window_size, step_size)
                
                # 添加到数据集
                self.X.extend(windows)
                self.y.extend([mode] * len(windows))
                
                mode_windows += len(windows)
                print(f"  ✓ {npy_file}: {len(windows)} 个窗口")
            
            total_windows += mode_windows
            print(f"  模式 {mode} 总计: {mode_windows} 个窗口")
            print()
        
        print(f"总计生成 {total_windows} 个训练样本")
        
        # 转换为numpy数组
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print(f"数据形状: X={self.X.shape}, y={self.y.shape}")
        
    def normalize_data(self):
        """标准化数据"""
        print("正在标准化数据...")
        
        # 将数据重塑为2D进行标准化
        original_shape = self.X.shape
        X_reshaped = self.X.reshape(-1, self.X.shape[-1])
        
        # 标准化
        X_normalized = self.scaler.fit_transform(X_reshaped)
        
        # 恢复原始形状
        self.X = X_normalized.reshape(original_shape)
        
        print("✓ 数据标准化完成")
    
    def shuffle_data(self):
        """打乱数据"""
        print("正在打乱数据...")
        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        print("✓ 数据打乱完成")
    
    def save_training_data(self, filename="training_data.npz"):
        """保存训练数据"""
        print(f"正在保存训练数据到 {filename}...")
        
        # 保存数据
        np.savez_compressed(filename, 
                          X=self.X, 
                          y=self.y,
                          mode_names=self.mode_names)
        
        # 保存标准化器
        scaler_filename = filename.replace('.npz', '_scaler.pkl')
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存数据集信息
        info = {
            "total_samples": len(self.X),
            "window_size": self.X.shape[1],
            "num_classes": len(self.mode_names),
            "class_distribution": {int(mode): int(np.sum(self.y == mode)) for mode in self.mode_names.keys()},
            "data_shape": self.X.shape,
            "mode_names": self.mode_names
        }
        
        info_filename = filename.replace('.npz', '_info.json')
        with open(info_filename, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ 训练数据已保存:")
        print(f"  - 数据文件: {filename}")
        print(f"  - 标准化器: {scaler_filename}")
        print(f"  - 信息文件: {info_filename}")
        
        return info
    
    def show_data_distribution(self):
        """显示数据分布"""
        print("\n数据分布:")
        print("-" * 30)
        
        for mode, mode_name in self.mode_names.items():
            count = np.sum(self.y == mode)
            percentage = count / len(self.y) * 100
            print(f"模式 {mode} ({mode_name}): {count} 样本 ({percentage:.1f}%)")
        
        print(f"\n总计: {len(self.y)} 样本")
    
    def generate_training_data(self, window_size=2000, step_size=200, 
                             normalize=True, shuffle_data=True, 
                             save_filename="training_data.npz"):
        """
        一键生成训练数据
        
        参数:
        - window_size: 窗口大小
        - step_size: 步长
        - normalize: 是否标准化
        - shuffle_data: 是否打乱
        - save_filename: 保存文件名
        """
        
        # 1. 加载并处理样本
        self.load_and_process_samples(window_size, step_size)
        
        if len(self.X) == 0:
            print("❌ 没有生成任何训练样本，请检查典型样本数据")
            return None
        
        # 2. 显示数据分布
        self.show_data_distribution()
        
        # 3. 标准化（可选）
        if normalize:
            self.normalize_data()
        
        # 4. 打乱数据（可选）
        if shuffle_data:
            self.shuffle_data()
        
        # 5. 保存数据
        info = self.save_training_data(save_filename)
        
        print("\n" + "="*50)
        print("🎉 训练数据生成完成!")
        print("="*50)
        
        return info

# ========== 使用示例 ==========
def main():
    # 创建生成器
    generator = TrainingDataGenerator()
    
    # 生成训练数据
    info = generator.generate_training_data(
        window_size=20000,      # 窗口大小：2000个数据点
        step_size=2000,         # 步长：200个数据点 (90%重叠)
        normalize=True,        # 标准化数据
        shuffle_data=True,     # 打乱数据
        save_filename="training_data.npz"  # 保存文件名
    )
    
    if info:
        print(f"\n生成的训练数据信息:")
        print(f"- 总样本数: {info['total_samples']}")
        print(f"- 窗口大小: {info['window_size']}")
        print(f"- 类别数: {info['num_classes']}")
        print(f"- 数据形状: {info['data_shape']}")

# ========== 数据加载示例 ==========
def load_training_data_example():
    """展示如何加载训练数据"""
    print("加载训练数据示例:")
    
    # 加载数据
    data = np.load('training_data.npz')
    X = data['X']
    y = data['y']
    mode_names = data['mode_names'].item()
    
    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")
    print(f"模式名称: {mode_names}")
    
    # 加载标准化器
    with open('training_data_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("✓ 训练数据加载完成")
    
    return X, y, mode_names, scaler

if __name__ == "__main__":
    main()
    
    # 如果需要测试加载
    # load_training_data_example()