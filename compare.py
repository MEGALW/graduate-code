import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# ==========================================
# 1. 导入模型核心 
# ==========================================
from train import AdaptiveInfraredUNet, Config, generate_adaptive_map

# ==========================================
# 2. 核心指标计算器 
# ==========================================
class ImageMetrics:
    @staticmethod
    def calculate_entropy(image):
        """计算信息熵 (Entropy) - 越大代表细节越多"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
        return entropy

    @staticmethod
    def calculate_average_gradient(image):
        """计算平均梯度 (AG) - 越大代表边缘越清晰"""
        img = image.astype(np.float32)
        dx = np.diff(img, axis=1) 
        dy = np.diff(img, axis=0) 
        dx = dx[:-1, :]
        dy = dy[:, :-1]
        grad = np.sqrt((dx**2 + dy**2) / 2.0)
        return np.mean(grad)

# ==========================================
# 3. 经典基线算法实现 
# ==========================================
class Baselines:
    @staticmethod
    def run_clahe(image_np):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image_np)

    @staticmethod
    def run_equalize_hist(image_np):
        return cv2.equalizeHist(image_np)

# ==========================================
# 4. 主程序：全景对比与诊断
# ==========================================
def main():
    print("🚀 启动红外增强全景对比与诊断系统...")
    
    cfg = Config()
    weight_path = os.path.join(cfg.save_dir, "./best_weight.pth") 
    
    # 【⚠️请核对这里的图片路径】
    low_img_path = "I:/data3/dataset/train/low/190001.png"   
    gt_img_path = "I:/data3/dataset/train/high/190001.jpg"    
    
    output_folder = "I:/data3/comparison_plots" 
    os.makedirs(output_folder, exist_ok=True)

    print(f"🖥️  正在连接计算设备: {cfg.device}")
    model = AdaptiveInfraredUNet().to(cfg.device)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=cfg.device))
        model.eval()
        print("✅ 巅峰权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    if not os.path.exists(low_img_path) or not os.path.exists(gt_img_path):
        print(f"❌ 图片不存在，请核对路径！")
        return

    img_low_np = cv2.imread(low_img_path, cv2.IMREAD_GRAYSCALE)
    img_gt_np = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
    original_h, original_w = img_low_np.shape
    
    valid_h = (original_h // 4) * 4
    valid_w = (original_w // 4) * 4
    img_valid = cv2.resize(img_low_np, (valid_w, valid_h))
    
    low_tensor = torch.from_numpy(img_valid).float().unsqueeze(0).unsqueeze(0) / 255.0
    low_tensor = low_tensor.to(cfg.device)

    print("\n⚔️  所有增强算法正在施展魔法...")
    
# 替换为这些
    from train import DiffusionPipeline  # 确保在文件开头导入了 Pipeline
    
    pipeline = DiffusionPipeline(cfg.timesteps, cfg.device)
    ada_map = generate_adaptive_map(low_tensor)
    
    with torch.no_grad():
        # 开启完整的 1000 步去噪魔法！
        enhanced_tensor = pipeline.p_sample_loop(model, low_tensor, ada_map)

    enhanced_ours_np = enhanced_tensor.squeeze().cpu().numpy()
    enhanced_ours_np = np.clip(enhanced_ours_np * 255.0, 0, 255).astype(np.uint8)
    enhanced_ours_np = cv2.resize(enhanced_ours_np, (original_w, original_h))

    enhanced_clahe_np = Baselines.run_clahe(img_low_np)
    enhanced_equ_np = Baselines.run_equalize_hist(img_low_np)

    print("📊 正在收集全网评分...")
    
    def get_full_metrics(pred_np, target_np):
        pred_valid = cv2.resize(pred_np, (original_w, original_h))
        target_valid = cv2.resize(target_np, (original_w, original_h)) 
        psnr = psnr_func(target_valid, pred_valid, data_range=255) 
        ssim = ssim_func(target_valid, pred_valid, data_range=255)
        en = ImageMetrics.calculate_entropy(pred_valid)
        ag = ImageMetrics.calculate_average_gradient(pred_valid)
        return psnr, ssim, en, ag

    psnr_low, ssim_low, en_low, ag_low = get_full_metrics(img_low_np, img_gt_np)
    psnr_ours, ssim_ours, en_ours, ag_ours = get_full_metrics(enhanced_ours_np, img_gt_np)
    psnr_clahe, ssim_clahe, en_clahe, ag_clahe = get_full_metrics(enhanced_clahe_np, img_gt_np)
    psnr_equ, ssim_equ, en_equ, ag_equ = get_full_metrics(enhanced_equ_np, img_gt_np)

    print("\n🎉 正在渲染防重叠排版的可视化界面...")
    
    # 【排版优化1】：加大整体画布，特别是让右侧面板更宽 (width_ratios调整为 2:1.2)
    fig = plt.figure(figsize=(22, 14), facecolor='white')
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1.2])
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 

    plot_titles = [
        f"高清 Ground Truth (完美参考图)\n[分辨率: {original_w}x{original_h}]",
        "低质量输入图 (模糊/含噪声)",
        "你的模型增强结果 (巅峰权重 30.62dB)"
    ]
    plot_images = [img_gt_np, img_low_np, enhanced_ours_np]

    for i in range(3):
        ax = plt.subplot(gs[i, 0])
        ax.imshow(plot_images[i], cmap='gray', vmin=0, vmax=255)
        ax.set_title(plot_titles[i], fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')

    # 【排版优化2】：为四列数据强制指定不会重叠的绝对 X 坐标
    col_x = [0.35, 0.55, 0.75, 0.95] 
    
    metric_labels = ['PSNR (dB) ↑', 'SSIM ↑', '信息熵 (EN) ↑', '平均梯度(AG) ↑']
    metric_low = [f"{psnr_low:.2f}", f"{ssim_low:.4f}", f"{en_low:.2f}", f"{ag_low:.2f}"]
    metric_ours = [f"{psnr_ours:.2f}", f"{ssim_ours:.4f}", f"{en_ours:.2f}", f"{ag_ours:.2f}"]
    metric_clahe = [f"{psnr_clahe:.2f}", f"{ssim_clahe:.4f}", f"{en_clahe:.2f}", f"{ag_clahe:.2f}"]
    metric_equ = [f"{psnr_equ:.2f}", f"{ssim_equ:.4f}", f"{en_equ:.2f}", f"{ag_equ:.2f}"]

    # 卡片1：GT
    ax_card_gt = plt.subplot(gs[0, 1])
    ax_card_gt.axis('off')
    ax_card_gt.text(0.5, 0.7, "💯 高清目标参考图\n全套指标作为理论满分基准", fontsize=20, 
                    ha='center', va='center', color='green', fontweight='bold',
                    bbox=dict(facecolor='#d4edda', edgecolor='none'))

    # 卡片2：基线数据
    ax_card_low = plt.subplot(gs[1, 1])
    ax_card_low.axis('off')
    
    row_labels = ["客观指标", "低质量原图", "CLAHE (传统)", "直方图均衡化"]
    y_start = 0.9
    ax_card_low.text(0.02, y_start, "📋 基准与传统算法报告", fontsize=18, fontweight='bold')
    y_start -= 0.2
    
    for i in range(len(row_labels)):
        color = '#333333' if i == 0 else '#555555'
        weight = 'bold' if i == 0 else 'normal'
        ax_card_low.text(0.02, y_start - i*0.18, row_labels[i], fontsize=15, color=color, fontweight=weight) # 增大行间距0.18
        
        for j in range(4):
            if i == 0:
                text, weight, fsize = metric_labels[j], 'bold', 14
            elif i == 1: text, fsize = metric_low[j], 14
            elif i == 2: text, fsize = metric_clahe[j], 14
            elif i == 3: text, fsize = metric_equ[j], 14
            
            # 使用绝对列坐标，防止数值变长挤压
            ax_card_low.text(col_x[j], y_start - i*0.18, text, fontsize=fsize, color=color, fontweight=weight, ha='center')

    # 卡片3：你的模型
    ax_card_ours = plt.subplot(gs[2, 1])
    ax_card_ours.axis('off')
    # 彻底移除了报错的 boxstyle
    rect = plt.Rectangle((0, 0), 1, 1, facecolor='#e3f2fd', edgecolor='none', transform=ax_card_ours.transAxes)
    ax_card_ours.add_patch(rect)
    
    y_start = 0.85
    ax_card_ours.text(0.02, y_start, "🌟 你的注意力 U-Net 测试报告", fontsize=18, fontweight='bold', color='#1565c0')
    y_start -= 0.25
    
    for j in range(4):
        ax_card_ours.text(col_x[j], y_start, metric_labels[j], fontsize=14, fontweight='bold', color='#1565c0', ha='center')
        
    y_start -= 0.2
    ax_card_ours.text(0.02, y_start, "自适应扩散系统", fontsize=16, fontweight='bold', color='#1565c0')
    for j in range(4):
        ax_card_ours.text(col_x[j], y_start, metric_ours[j], fontsize=18, fontweight='bold', color='#1565c0', ha='center')

    psnr_imp = (psnr_ours - psnr_low) if psnr_low > 0 else 0
    ax_card_ours.text(0.5, 0.15, f"💯 结论：相比原图，PSNR 净提升 {psnr_imp:.2f} dB！", 
                      fontsize=17, fontweight='bold', color='#1565c0', ha='center', va='center',
                      bbox=dict(facecolor='white', edgecolor='#90caf9'))

    save_path = os.path.join(output_folder, "panoramic_comparison.png")
    
    # 【新增双保险】：无论文件夹存不存在，保存前强行沿途创建好所有必需的文件夹！
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    main()