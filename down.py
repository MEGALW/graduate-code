import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# ==========================================
# 红外图像退化算子定义
# ==========================================

def reduce_contrast(img, alpha_range=(0.4, 0.7)):
    """
    1. 降低对比度：模拟极低温差环境（目标与背景温度接近）
    alpha 越小，图像越灰蒙蒙
    """
    alpha = np.random.uniform(*alpha_range)
    # 将图像整体亮度向中性灰 (128) 靠拢
    degraded = img.astype(np.float32)
    degraded = 128.0 + alpha * (degraded - 128.0)
    return np.clip(degraded, 0, 255).astype(np.uint8)

def add_thermal_blur(img, max_kernel_size=5):
    """
    2. 模拟热晕/光学散焦：高频边缘模糊
    """
    kernel_size = np.random.choice(range(1, max_kernel_size + 1, 2))
    if kernel_size > 1:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def add_gaussian_noise(img, mean=0, var_range=(10, 50)):
    """
    3. 添加高斯噪声：模拟红外传感器的本底热噪声
    """
    var = np.random.uniform(*var_range)
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape)
    
    degraded = img.astype(np.float32) + gaussian
    return np.clip(degraded, 0, 255).astype(np.uint8)

def add_stripe_noise(img, stripe_prob=0.3, intensity_range=(5, 20)):
    """
    4. 添加条纹噪声：模拟红外非均匀性校正 (NUC) 误差
    """
    if np.random.rand() > stripe_prob:
        return img 
        
    degraded = img.astype(np.float32)
    h, w = img.shape
    
    intensity = np.random.uniform(*intensity_range)
    if np.random.rand() > 0.5:
        # 垂直条纹（Numpy 的默认广播规则可以直接匹配，不会报错）
        num_stripes = np.random.randint(5, 20)
        cols = np.random.choice(w, num_stripes, replace=False)
        degraded[:, cols] += np.random.choice([-intensity, intensity], num_stripes)
    else:
        # 水平条纹（需要重塑维度以支持广播）
        num_stripes = np.random.randint(5, 20)
        rows = np.random.choice(h, num_stripes, replace=False)
        
        # 【修改这里👇】加上 .reshape(-1, 1) 将其变成 (18, 1) 的形状
        noise_values = np.random.choice([-intensity, intensity], num_stripes).reshape(-1, 1)
        degraded[rows, :] += noise_values
        
    return np.clip(degraded, 0, 255).astype(np.uint8)
def process_image(img):
    """流水线：按顺序执行降质"""
    img = reduce_contrast(img)
    img = add_thermal_blur(img)
    img = add_stripe_noise(img)
    img = add_gaussian_noise(img)
    return img

# ==========================================
# 主执行流程
# ==========================================

def create_degraded_dataset(high_dir, low_dir):
    """
    读取 high 目录下的原图，降质后保存到 low 目录
    """
    os.makedirs(low_dir, exist_ok=True)
    
    # 获取所有图片路径 (LLVIP 通常是 .jpg 或 .png)
    image_paths = glob.glob(os.path.join(high_dir, "*.jpg")) + \
                  glob.glob(os.path.join(high_dir, "*.png"))
    
    if not image_paths:
        print(f"❌ 在 {high_dir} 中没有找到图片，请检查路径！")
        return

    print(f"🚀 开始生成配对数据集，共找到 {len(image_paths)} 张图像...")
    
    for img_path in tqdm(image_paths, desc="降质处理中"):
        # 1. 读取高质图 (灰度模式)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 2. 进行退化处理
        degraded_img = process_image(img)
        
        # 3. 保存到 low 目录 (保持文件名完全一致)
        filename = os.path.basename(img_path)
        save_path = os.path.join(low_dir, filename)
        
        # 强制保存为 PNG 以避免 JPEG 再次压缩带来不可控的伪影
        save_path = os.path.splitext(save_path)[0] + '.png'
        cv2.imwrite(save_path, degraded_img)

if __name__ == "__main__":
    # 【注意】请将下面的路径替换为你电脑上的实际路径
    # 假设你已经把挑选出来的 LLVIP 红外图放到了这个目录
    SOURCE_HIGH_DIR = "I:/data2/dataset/test/high" 
    
    # 脚本会自动创建这个目录，并把生成的渣画质图放进去
    TARGET_LOW_DIR = "I:/data2/dataset/test/low"   
    
    create_degraded_dataset(SOURCE_HIGH_DIR, TARGET_LOW_DIR)
    print("✅ 数据集降质合成完毕！现在你的扩散模型有事可做了。")