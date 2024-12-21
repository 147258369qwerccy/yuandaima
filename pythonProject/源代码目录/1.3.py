import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("Error: Could not load image.")
    exit()

# 应用直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 计算并绘制原图直方图
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

plt.figure(figsize=(12, 6))

# 绘制原图直方图
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.bar(bins[:-1], hist, width=1, edgecolor='black')
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# 计算并绘制均衡化后的直方图
equalized_hist, bins = np.histogram(equalized_image.flatten(), 256, [0, 256])

# 绘制均衡化后的图像
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# 绘制均衡化后的直方图
plt.subplot(2, 2, 4)
plt.bar(bins[:-1], equalized_hist, width=1, edgecolor='black')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

