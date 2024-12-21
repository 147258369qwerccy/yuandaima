import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print("Error: Could not load image.")
    exit()

# 但这里我们是对RGB每个通道分别进行均衡化，仅作为示例
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # 仅对V通道均衡化（这不是标准做法）
# 但为了展示，我们分别对R、G、B通道进行均衡化
r_channel, g_channel, b_channel = cv2.split(image)
r_equalized = cv2.equalizeHist(r_channel)
g_equalized = cv2.equalizeHist(g_channel)
b_equalized = cv2.equalizeHist(b_channel)
equalized_image = cv2.merge([r_equalized, g_equalized, b_equalized])


# 绘制原图RGB直方图
def plot_hist(image, title):
    colors = ('r', 'g', 'b')
    channel_hist = []
    for (channel, color) in zip(cv2.split(image), colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        channel_hist.append(hist)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')


# 显示原图及其RGB直方图
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB以正确显示颜色
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plot_hist(image, 'Histogram of Original Image')

# 显示均衡化后的图像及其RGB直方图
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))  # 转换为RGB以正确显示颜色
plt.title('Equalized Image (Each Channel)')
plt.axis('off')

plt.subplot(2, 2, 4)
plot_hist(equalized_image, 'Histogram of Equalized Image (Each Channel)')

plt.tight_layout()
plt.show()


