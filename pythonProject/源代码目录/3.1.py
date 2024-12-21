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

# 对结果进行归一化，使其适应显示
gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))

# 显示原图和边缘检测结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
# 应用高斯滤波进行平滑
smoothed_image = cv2.GaussianBlur(image, (5, 5), 1.5)

# 使用平滑后的图像进行Prewitt边缘检测
smoothed_gradient_x = cv2.filter2D(smoothed_image, -1, prewitt_kernel_x)
smoothed_gradient_y = cv2.filter2D(smoothed_image, -1, prewitt_kernel_y)

# 计算梯度幅度
smoothed_gradient_magnitude = np.sqrt(smoothed_gradient_x**2 + smoothed_gradient_y**2)
smoothed_gradient_magnitude = np.uint8(np.clip(smoothed_gradient_magnitude, 0, 255))

# 显示平滑后的边缘检测结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_gradient_magnitude, cmap='gray')
plt.title('Smoothed Prewitt Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
# 使用Otsu方法进行阈值化
_, thresholded_image = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示阈值化后的结果
plt.figure(figsize=(6, 6))
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Prewitt Edge Detection')
plt.axis('off')
plt.show()
