import os
import cv2
import torch
import glob
import numpy as np
from tqdm import tqdm

# 从你的主训练文件中导入模型架构和参数
from train import AdaptiveInfraredUNet, DiffusionPipeline, Config, generate_adaptive_map

def main():
    print("🚀 启动深度学习增强算法 (自适应注意力扩散模型)...")
    
    # 1. 设置路径配置
    input_dir = "./test/in"
    output_dir = "./test/out"
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = Config()
    weight_path = os.path.join(cfg.save_dir, "./best_weight.pth")
    
    # 2. 初始化模型和扩散管道
    print(f"🖥️ 正在连接计算设备: {cfg.device}")
    model = AdaptiveInfraredUNet().to(cfg.device)
    pipeline = DiffusionPipeline(cfg.timesteps, cfg.device)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=cfg.device))
        model.eval()
        print("✅ 巅峰权重 (32.11dB) 加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败，请检查路径: {e}")
        return

    # 3. 寻找图片
    image_paths = glob.glob(os.path.join(input_dir, "*.*"))
    if not image_paths:
        print(f"❌ 在 {input_dir} 下没有找到图片！")
        return
        
    print(f"📂 找到 {len(image_paths)} 张图片，准备进行高清重构...")
    
    # 4. 批量执行 1000 步去噪推理
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"\n正在处理: {filename}")
        
        # 读取原图并转换尺寸以适配网络结构 (必须是 4 的倍数)
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        original_h, original_w = img_np.shape
        valid_h, valid_w = (original_h // 4) * 4, (original_w // 4) * 4
        img_valid = cv2.resize(img_np, (valid_w, valid_h))
        
        # 转换为 Tensor 送入显卡
        low_tensor = torch.from_numpy(img_valid).float().unsqueeze(0).unsqueeze(0) / 255.0
        low_tensor = low_tensor.to(cfg.device)
        
        with torch.no_grad():
            # 生成自适应权重图
            ada_map = generate_adaptive_map(low_tensor)
            # 执行扩散去噪
            enhanced_tensor = pipeline.p_sample_loop(model, low_tensor, ada_map)
        
        # 将 Tensor 转换回图片格式
        enhanced_np = enhanced_tensor.squeeze().cpu().numpy()
        enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
        # 还原回原始分辨率
        enhanced_np = cv2.resize(enhanced_np, (original_w, original_h))
        
        # 保存结果
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, enhanced_np)
        print(f"  -> 已保存至: {save_path}")

    print(f"\n🎉 深度学习批量增强完毕！完美收工！")

if __name__ == "__main__":
    main()