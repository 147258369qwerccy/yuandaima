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

    # 进行离散傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 将零频率分量移动到频谱中心

    # 计算频谱的幅度谱（magnitude spectrum）
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 进行逆离散傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)  # 取绝对值，因为结果可能是复数

    img_back = np.uint8(np.clip(img_back, 0, 255))

    # 显示原始图像、幅度谱和逆变换后的图像
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Magnitude Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Image after IDFT')
    plt.imshow(img_back, cmap='gray')

    plt.show()
