import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

# 使用拉普拉斯算子进行边缘检测
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# 显示边缘检测结果
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Edge Detection')
plt.show()
# 设置阈值 T
T = 50  # 可以根据需要调整此值

# 二值化处理，生成二值图像 g_T(x, y)
_, binary_edge = cv2.threshold(laplacian, T, 255, cv2.THRESH_BINARY)

# 显示二值化结果
plt.imshow(binary_edge, cmap='gray')
plt.title('Binary Edge Image')
plt.show()
# 获取原图像中对应于边缘位置的像素值
edge_pixels = image[binary_edge == 255]

# 计算直方图
hist, bins = np.histogram(edge_pixels, bins=256, range=(0, 256))

# 显示直方图
plt.plot(bins[:-1], hist)
plt.title('Histogram of Edge Pixels')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# 使用Otsu方法进行全局阈值分割
# Otsu方法返回一个自动计算的阈值
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示Otsu分割结果
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu Segmentation Result')
plt.show()


