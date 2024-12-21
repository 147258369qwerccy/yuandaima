import cv2
import numpy as np
import matplotlib.pyplot as plt


def WeiPingMianQiePian():
    # 读取灰度图像
    img = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

    # 存放8个位平面
    bit_planes = []

    # 遍历从0到7的每个位
    for i in range(8):
        # 提取每个位平面，并将0和1转换为255和0
        bit_plane = (img >> i) & 1
        bit_plane = bit_plane * 255  # 显示为黑白图像
        bit_planes.append(bit_plane)

    # 创建显示的图像窗口
    plt.figure(figsize=(10, 6))

    # 显示每个位平面
    for i in range(8):
        plt.subplot(2, 4, i + 1)  # 创建2行4列的子图
        plt.imshow(bit_planes[i], cmap='gray')  # 显示位平面
        plt.title(f'Bit Plane {i}')  # 设置标题
        plt.axis('off')  # 隐藏坐标轴

    # 调整布局以避免重叠
    plt.tight_layout()
    # 显示图像
    plt.show()

# 调用函数
WeiPingMianQiePian()
