import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image_path = 'tuxi4.jpg'   # 替换为你的图像路径
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

    # 定义巴特沃斯低通滤波器的参数
    D0 = 30  # 截止频率（距离频谱中心的距离阈值）
    n = 2  # 滤波器阶数（决定了过渡带的陡峭程度）

    rows_half, cols_half = rows // 2, cols // 2
    butterworth_kernel = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到频谱中心的距离（归一化到[0, 1]范围）
            distance = np.sqrt((i - rows_half) ** 2 + (j - cols_half) ** 2) / np.sqrt(rows_half ** 2 + cols_half ** 2)
            # 应用巴特沃斯函数计算频率响应
            butterworth_kernel[i, j] = 1 / (1 + (distance / D0) ** (2 * n))

            # 应用滤波器到频域图像上
    fshift_filtered = fshift * butterworth_kernel

    # 进行逆离散傅里叶变换（IDFT）
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)  # 取绝对值，因为结果可能是复数


    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back_uint8 = np.uint8(img_back_normalized)

    # 显示原始图像和滤波后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Image after Butterworth Low-Pass Filtering')
    plt.imshow(img_back_uint8, cmap='gray')

    plt.show()
