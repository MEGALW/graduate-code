import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 读取图像并转换为 numpy 数组 (height, width, 3), dtype=uint8
im = np.array(Image.open('./1.jpg'))
# 设置噪声参数(可以改)
sigma = 50
um = 20
# 生成高斯噪声 (与 im 形状相同，均值为0)
noise = np.random.normal(um, sigma, im.shape)
# 添加噪声并裁剪到有效范围，再转为 uint8
im_noisy = im + noise
im_noisy = np.clip(im_noisy, 0, 255).astype(np.uint8)
# 显示图像
plt.imshow(im_noisy)
plt.axis('off')   # 不显示坐标轴
plt.show()