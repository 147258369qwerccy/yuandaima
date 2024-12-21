import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# 读取图像并转换为灰度图
img = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

# 显示原始图像
plt.figure(figsize=(6, 6))
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# 计算HOG特征
# block_norm='L2-Hys' 指定了HOG的归一化方式
features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

# 归一化的HOG图像
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 显示HOG图像
plt.figure(figsize=(6, 6))
plt.title('HOG Image')
plt.imshow(hog_image_rescaled, cmap='gray')
plt.axis('off')
plt.show()

# 计算HOG特征的直方图
# 获取每个cell内的梯度方向直方图
# HOG特征已经自动计算了梯度直方图，我们可以直接使用返回的 features 进行可视化
plt.figure(figsize=(8, 6))
plt.title('HOG Feature Histogram')
plt.hist(features, bins=50, range=(0, 1), color='gray')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.show()


