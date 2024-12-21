import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('tuxi4.jpg')

# 将BGR图片转换为RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 分离RGB分量
r, g, b = cv2.split(image_rgb)

# 将图片从RGB转换为HSI
image_hsi = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)  # OpenCV使用HSV而不是HSI，但原理相似
h, s, v = cv2.split(image_hsi)

# 由于OpenCV的HSV中的H分量是0-179，我们可以将其标准化到0-255以便显示
h_normalized = (h / 180.0 * 255).astype(np.uint8)

# 创建一个包含所有分量图的图像网格
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 显示RGB分量
axes[0, 0].imshow(r, cmap='gray')
axes[0, 0].set_title('R Channel')
axes[0, 1].imshow(g, cmap='gray')
axes[0, 1].set_title('G Channel')
axes[0, 2].imshow(b, cmap='gray')
axes[0, 2].set_title('B Channel')

# 显示HSI分量（使用HSV代替）
axes[1, 0].imshow(h_normalized, cmap='hsv')  # 使用hsv colormap来显示色调
axes[1, 0].set_title('H Channel (normalized)')
axes[1, 1].imshow(s, cmap='gray')
axes[1, 1].set_title('S Channel')
axes[1, 2].imshow(v, cmap='gray')
axes[1, 2].set_title('V Channel')

# 调整布局并显示图像
plt.tight_layout()
plt.show()

