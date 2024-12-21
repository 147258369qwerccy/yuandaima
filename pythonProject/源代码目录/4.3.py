import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
img = cv2.imread('tuxi4.jpg')

# 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为浮动类型
gray = np.float32(gray)

# 使用Harris角点检测
# 参数：dst是角点检测的结果图，blockSize为邻域的大小，ksize为Sobol滤波器的大小，k为Harris检测的参数
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# 结果进行膨胀处理，便于显示
dst = cv2.dilate(dst, None)

# 在图像中标记角点，标记的颜色为红色
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 可以调整阈值来检测不同的角点数量

# 显示结果
plt.figure(figsize=(6, 6))
plt.title('Harris Corner Detection')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
