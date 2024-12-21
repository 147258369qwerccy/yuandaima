import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像并转换为灰度图像
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 定义Prewitt算子的水平和垂直卷积核
prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

# 使用Prewitt算子进行卷积计算
gradient_x = cv2.filter2D(image, -1, prewitt_kernel_x)
gradient_y = cv2.filter2D(image, -1, prewitt_kernel_y)

# 计算梯度幅度
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))

# 使用Canny边缘检测
canny_edges = cv2.Canny(image, threshold1=100, threshold2=200)

# 显示原图、Prewitt边缘检测结果和Canny边缘检测结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()

