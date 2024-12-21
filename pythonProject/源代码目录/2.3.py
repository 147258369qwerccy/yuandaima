import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift
from skimage.restoration import wiener
import matplotlib.pyplot as plt


# 生成运动模糊核函数
def motion_blur_kernel(size, angle):
    kernel = np.zeros((size, size))
    center = size // 2
    angle = np.deg2rad(angle)
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    for i in range(size):
        offset = int((i - center) * sin_val / cos_val)
        if 0 <= center + offset < size:
            kernel[i, center + offset] = 1
    kernel /= kernel.sum()
    return kernel


# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=0.000001):
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


# 约束最小二乘滤波
def constrained_least_squares_filter(image, kernel, lambd=0.000001):
    kernel_fft = fft2(kernel, s=image.shape)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_fft = fft2(laplacian, s=image.shape)

    numerator = np.conj(kernel_fft) * fft2(image)
    denominator = (np.abs(kernel_fft) ** 2 + lambd * np.abs(laplacian_fft) ** 2)

    restored_fft = numerator / denominator
    restored_image = np.real(ifft2(restored_fft))
    return np.clip(restored_image, 0, 1)


# 读取并预处理图像
image = cv2.imread(cv2.samples.findFile('tuxi4.jpg'), cv2.IMREAD_GRAYSCALE)
image = image / 255.0

# 添加运动模糊和噪声
motion_kernel = motion_blur_kernel(15, 30)
blurred_image = convolve2d(image, motion_kernel, boundary='wrap', mode='same')
noisy_blurred_image = add_gaussian_noise(blurred_image, mean=0, var=0.000001)

# 使用维纳滤波恢复图像
restored_wiener = wiener(noisy_blurred_image, motion_kernel, balance=0.000001)

# 使用约束最小二乘法滤波恢复图像
restored_cls = constrained_least_squares_filter(noisy_blurred_image, motion_kernel, lambd=0.000001)

# 显示图像
plt.figure(figsize=(12, 8))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 4, 2), plt.imshow(noisy_blurred_image, cmap='gray'), plt.title('Blurred and Noisy')
plt.axis('off')
plt.subplot(1, 4, 3), plt.imshow(restored_wiener, cmap='gray'), plt.title('Wiener Filter')
plt.axis('off')
plt.subplot(1, 4, 4), plt.imshow(restored_cls, cmap='gray'), plt.title('CLS Filter')
plt.axis('off')
plt.show()
