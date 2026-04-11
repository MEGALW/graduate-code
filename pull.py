import os
import torch
import torch_directml
# 从你刚才的训练脚本里导入原有的类（假设你的原文件叫 train.py）
from train import AdaptiveInfraredUNet, Config, Evaluator

print("🚀 正在读取已训练好的模型权重...")
cfg = Config()

# 1. 实例化一个空模型并放到设备上
model = AdaptiveInfraredUNet().to(cfg.device)

# 2. 找到你刚训练好的权重文件
weight_path = os.path.join(cfg.save_dir, "best_weight.pth")

# 3. 把训练好的灵魂（权重）注入到空壳模型里
model.load_state_dict(torch.load(weight_path, map_location=cfg.device))
print("✅ 权重加载成功！")

# 4. 执行导出
onnx_path = os.path.join(cfg.save_dir, "adaptive_infrared_enhancer.onnx")
Evaluator.export_deployment_model(model, onnx_path)