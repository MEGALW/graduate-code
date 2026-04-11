import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch_directml

# 【重要】从你之前的训练代码文件中导入模型结构
# 假设你之前的训练代码文件叫 train_amd_infrared.py
# 如果你的文件名不同，请把 train_amd_infrared 改成你的实际文件名（不要加 .py）
from train import AdaptiveInfraredUNet, DiffusionPipeline, Config, generate_adaptive_map

def main():
    print("🚀 启动红外图像批量增强系统...")
    
    # ==========================================
    # 1. 路径与配置设定 (请根据你的实际情况修改)
    # ==========================================
    cfg = Config()
    
    # 你的最优权重文件路径
    weight_path = os.path.join(cfg.save_dir, "I:/outputs/best_weight.pth")
    
    # 待处理的低质量图像文件夹 (你需要提前建好并放几张测试图进去)
    input_folder = "I:/data3/dataset/train/low" 
    
    # 处理后输出的高质量图像文件夹
    output_folder = "I:/data3/dataset/train/enhanced_output" 
    os.makedirs(output_folder, exist_ok=True)

    # ==========================================
    # 2. 唤醒 AMD 显卡并加载“大脑”
    # ==========================================
    print(f"🖥️  正在连接计算设备: {cfg.device}")
    model = AdaptiveInfraredUNet().to(cfg.device)
    pipeline = DiffusionPipeline(cfg.timesteps, cfg.device)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=cfg.device))
        model.eval() # 开启推理模式（关闭 Dropout 等训练专属机制）
        print("✅ 巅峰权重 (30.62dB) 加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败，请检查路径: {e}")
        return

    # ==========================================
    # 3. 搜索待处理的图像
    # ==========================================
    image_paths = sorted(glob.glob(os.path.join(input_folder, "*.*")))
    if not image_paths:
        print(f"⚠️ 在 {input_folder} 文件夹中没有找到任何图片！")
        return
    
    print(f"📂 发现 {len(image_paths)} 张待处理图像，开始施展魔法...")

    # ==========================================
    # 4. 开始批量去噪增强 (核心循环)
    # ==========================================
    # 禁用梯度计算，节省显存并提速
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="整体处理进度"):
            # 提取文件名 (例如: 0001.png)
            filename = os.path.basename(img_path)
            
            # --- 步骤 A：读取图像（保持原汁原味的高清分辨率！） ---
            img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_np is None:
                continue
                
            original_h, original_w = img_np.shape
            
            # 【关键修复】：U-Net 下采样了两次，所以宽和高必须是 4 的倍数
            # 我们只做微调（比如把 1024 保持 1024，1281 变成 1280），绝不暴力压缩！
            valid_h = (original_h // 4) * 4
            valid_w = (original_w // 4) * 4
            img_valid = cv2.resize(img_np, (valid_w, valid_h))
            
            # 转换为 Tensor，此时送进模型的是高清大图
            low_tensor = torch.from_numpy(img_valid).float().unsqueeze(0).unsqueeze(0) / 255.0
            low_tensor = low_tensor.to(cfg.device)
            
            # --- 步骤 B：生成自适应权重图 ---
            ada_map = generate_adaptive_map(low_tensor)
            
            # --- 步骤 C：执行逆向去噪 (原分辨率直接运算) ---
            # 此时显卡是在运算上百万个像素，而不是可怜的 256x256
            enhanced_tensor = pipeline.p_sample_loop(model, low_tensor, ada_map)
            
            # --- 步骤 D：后处理与保存 ---
            enhanced_img = enhanced_tensor.squeeze().cpu().numpy()
            enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
            
            # 恢复最原始的精确尺寸（只拉伸那几个像素的误差，肉眼不可见）
            enhanced_img = cv2.resize(enhanced_img, (original_w, original_h))
            
            # 保存到输出文件夹
            save_path = os.path.join(output_folder, f"enhanced_{filename}")
            cv2.imwrite(save_path, enhanced_img)


    print(f"\n🎉 全部处理完毕！快去 {output_folder} 见证奇迹吧！")

if __name__ == "__main__":
    main()