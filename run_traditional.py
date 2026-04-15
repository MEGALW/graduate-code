import os
import cv2
import glob
from tqdm import tqdm

def main():
    print("⚙️ 启动传统红外图像增强算法 (CLAHE)...")
    
    # 1. 设置输入输出路径 (请根据你的实际情况修改)
    input_dir = "I:/graduate/test_images"
    output_dir = "I:/graduate/output_clahe"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 找到所有图片
    image_paths = glob.glob(os.path.join(input_dir, "*.*"))
    if not image_paths:
        print(f"❌ 在 {input_dir} 下没有找到图片！")
        return
        
    print(f"📂 找到 {len(image_paths)} 张图片，开始处理...")
    
    # 3. 初始化 CLAHE 算法 (限制对比度自适应直方图均衡化)
    # clipLimit 决定了对比度放大的上限，tileGridSize 决定了局部块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 4. 批量处理
    for img_path in tqdm(image_paths, desc="处理进度"):
        # 以灰度模式读取图片
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 应用 CLAHE 算法
        enhanced_img = clahe.apply(img)
        
        # 提取文件名并保存
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, enhanced_img)
        
    print(f"\n✅ 传统算法处理完毕！结果已保存至: {output_dir}")

if __name__ == "__main__":
    main()