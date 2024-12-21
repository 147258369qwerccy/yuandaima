import cv2
import numpy as np


def apply_mean_filter(image, kernel_size=3):

    # 使用OpenCV的blur函数应用均值滤波器
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))

    return filtered_image


# 读取图像
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print("Error: Unable to load image.")
else:
    # 判断图像是否为灰度图，如果不是则转换为灰度图（可选）
    if len(image.shape) == 3:  # 彩色图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Processing grayscale image.")
    else:  # 灰度图像
        gray_image = image
        print("Processing color image in grayscale mode.")

        # 应用均值滤波器
    kernel_size = 5  # 可以根据需要调整滤波器核的大小
    filtered_image = apply_mean_filter(gray_image, kernel_size)

    # 显示原始图像和滤波后的图像
    cv2.imshow('Original Image', gray_image)
    cv2.imshow('Filtered Image (Mean Filter)', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

