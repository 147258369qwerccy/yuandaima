import cv2
import numpy as np


def GrayLevelSliceEnhanceAndSmooth(image_path):
    # 读取灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 定义参数
    lower_bound = 50
    upper_bound = 200
    enhancement_factor = 1.5
    suppression_factor = 0.7

    # 创建图像副本用于增强
    enhanced_image = image.copy()

    # 灰度切片操作
    for i in range(image.shape[0]):  # 遍历行
        for j in range(image.shape[1]):  # 遍历列
            pixel_value = image[i, j]

            if lower_bound <= pixel_value <= upper_bound:
                # 对位于灰度切片范围内的像素增强
                enhanced_image[i, j] = np.clip(int(pixel_value * enhancement_factor), 0, 255)
            else:
                # 对其他像素进行压制
                enhanced_image[i, j] = np.clip(int(pixel_value * suppression_factor), 0, 255)

    # 应用高斯平滑滤波器
    kernel_size = (5, 5)  # 高斯核大小
    sigma = 1.0  # 标准差
    smoothed_image = cv2.GaussianBlur(enhanced_image, kernel_size, sigma)

    # 显示图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced and Smoothed Image', smoothed_image)

    # 等待键盘输入后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 调用函数
GrayLevelSliceEnhanceAndSmooth('tuxi4')


