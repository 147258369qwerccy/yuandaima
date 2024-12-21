import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 读取图像并转换为灰度图
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("Error: Unable to load image.")
else:
    # 获取图像的尺寸
    rows, cols = image.shape

    # 计算离散傅里叶变换（DFT）
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 将零频率分量移动到频谱中心

    # 定义高斯低通滤波器的标准差（控制滤波器的宽度）
    sigma = 30  # 你可以根据需要调整这个值

    # 创建高斯低通滤波器的频率响应
    rows_half, cols_half = rows // 2, cols // 2
    gaussian_kernel = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到频谱中心的距离（归一化到[0, 1]范围）
            distance = np.sqrt((i - rows_half) ** 2 + (j - cols_half) ** 2) / np.sqrt(rows_half ** 2 + cols_half ** 2)
            # 应用高斯函数计算频率响应
            gaussian_kernel[i, j] = np.exp(-(distance ** 2) / (2 * sigma ** 2))

            # 应用滤波器到频域图像上
    fshift_filtered = fshift * gaussian_kernel

    # 进行逆离散傅里叶变换（IDFT）
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)  # 取绝对值，因为结果可能是复数


    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back_uint8 = np.uint8(img_back_normalized)

    sigma_spatial = sigma / np.sqrt(2 * np.log(2))  # 将频域sigma转换为空间域sigma（基于高斯函数的等价性）

    img_spatial_filtered = cv2.GaussianBlur(image, (0, 0), sigma_spatial)  # (0, 0)表示根据sigma计算核大小

    # 显示原始图像、频域高斯滤波后的图像和空间域高斯滤波后的图像（作为对比）
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Image after Frequency Domain Gaussian Low-Pass Filtering')
    plt.imshow(img_back_uint8, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Image after Spatial Domain Gaussian Filtering (for Comparison)')
    plt.imshow(img_spatial_filtered, cmap='gray')

    plt.show()
