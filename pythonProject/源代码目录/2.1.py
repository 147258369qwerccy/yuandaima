import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_uniform_noise(image, low=-50, high=50):
    noise = np.random.uniform(low, high, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Add Salt
    salt_coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255

    # Add Pepper
    pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0

    return noisy_image


def plot_image_and_histogram(image, title, ax_img, ax_hist):
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')

    ax_hist.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
    ax_hist.set_title(f'Histogram of {title}')
    ax_hist.set_xlabel('Pixel Intensity')
    ax_hist.set_ylabel('Frequency')


# 读取灰度图像
image = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

# 添加噪声
gaussian_noise_img = add_gaussian_noise(image)
uniform_noise_img = add_uniform_noise(image)
salt_pepper_noise_img = add_salt_pepper_noise(image)

# 创建画布并展示原图和噪声图像以及它们的直方图
fig, axs = plt.subplots(4, 2, figsize=(12, 14))

# 原图和直方图
plot_image_and_histogram(image, "Original Image", axs[0, 0], axs[0, 1])

# 高斯噪声图像和直方图
plot_image_and_histogram(gaussian_noise_img, "Gaussian Noise Image", axs[1, 0], axs[1, 1])

# 均匀噪声图像和直方图
plot_image_and_histogram(uniform_noise_img, "Uniform Noise Image", axs[2, 0], axs[2, 1])

# 椒盐噪声图像和直方图
plot_image_and_histogram(salt_pepper_noise_img, "Salt & Pepper Noise Image", axs[3, 0], axs[3, 1])

# 调整布局
plt.tight_layout()
plt.show()
