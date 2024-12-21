import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # 定义理想低通滤波器的截止频率（以像素为单位）
    D0 = 30  # 你可以根据需要调整这个值

    rows_half, cols_half = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到频谱中心的距离
            distance = np.sqrt((i - rows_half) ** 2 + (j - cols_half) ** 2)
            # 应用理想低通滤波器的条件
            if distance <= D0:
                mask[i, j] = 1

                # 应用滤波器掩膜到频域图像上
    fshift_filtered = fshift * mask

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
    plt.title('Image after Ideal Low-Pass Filtering')
    plt.imshow(img_back_uint8, cmap='gray')

    plt.show()
