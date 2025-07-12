import numpy as np
import os
import json

class SampleCollector:
    """收集典型样本的工具"""
    
    def __init__(self, sample_rate=1e5):
        self.sample_rate = sample_rate
        self.base_dir = "typical_samples"
        self.mode_names = {
            0: "mode0_stable",
            1: "mode1_intermittent", 
            2: "mode2_limit_cycle",
            3: "mode3_beat"
        }
        
        # 创建文件夹
        for mode, name in self.mode_names.items():
            os.makedirs(os.path.join(self.base_dir, name), exist_ok=True)
    
    def collect_sample(self, data, file_id, start_time, end_time, mode):
        """
        收集一个样本
        
        参数:
        - data: processed_data.npy的数据
        - file_id: 文件标识符(如'0714')
        - start_time: 开始时间(秒)
        - end_time: 结束时间(秒)
        - mode: 模式(0/1/2/3)
        """
        
        # 提取数据段
        start_idx = int(start_time * self.sample_rate)
        end_idx = int(end_time * self.sample_rate)
        segment = data[start_idx:end_idx]
        
        # 保存路径
        mode_dir = os.path.join(self.base_dir, self.mode_names[mode])
        
        # 生成文件名
        sample_name = f"{file_id}_{start_time}_{end_time}s"
        data_file = os.path.join(mode_dir, f"{sample_name}.npy")
        info_file = os.path.join(mode_dir, f"{sample_name}.json")
        
        # 保存数据
        np.save(data_file, segment)
        
        # 保存信息
        info = {
            "file_id": file_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "mode": mode,
            "sample_rate": self.sample_rate,
            "data_points": len(segment)
        }
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ 已保存: {sample_name} → 模式{mode}")
        
        # 统计当前数量
        self.show_statistics()
    
    def show_statistics(self):
        """显示当前收集情况"""
        print("\n当前收集情况:")
        for mode, name in self.mode_names.items():
            mode_dir = os.path.join(self.base_dir, name)
            count = len([f for f in os.listdir(mode_dir) if f.endswith('.npy')])
            print(f"  模式{mode}: {count}个样本")
        print("")

# ========== 使用示例 ==========
def main():
    # 加载数据
    data = np.load('processed_data.npy')
    
    # 创建收集器
    collector = SampleCollector()
    
    # ===== 在这里添加您的样本 =====
    # 格式: collector.collect_sample(data, 文件ID, 开始时间, 结束时间, 模式)
    
    # 示例:
    collector.collect_sample(data, "0702", 1.4, 2.0, 3)  
   


 
    
    # 继续添加更多...
    # collector.collect_sample(data, "xxxx", x.x, x.x, x)
    
if __name__ == "__main__":
    main()