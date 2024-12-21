import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像（假设图像已经是灰度图）
img = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

# 显示原始图像
plt.figure(figsize=(6, 6))
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# 步骤1：使用Canny边缘检测
edges = cv2.Canny(img, 50, 150)

# 显示边缘检测后的结果
plt.figure(figsize=(6, 6))
plt.title('Edge Detection using Canny')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

# 步骤2：使用Hough变换检测直线
# cv2.HoughLines()返回的是直线的参数(rho, theta)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # 150是阈值

# 将检测到的直线画到图像上
img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 将灰度图转换为彩色图，以便绘制直线
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 画红色的直线

# 显示带有直线的图像
plt.figure(figsize=(6, 6))
plt.title('Detected Lines using Hough Transform')
plt.imshow(img_with_lines)
plt.axis('off')
plt.show()



