import cv2
import numpy as np
import matplotlib.pyplot as plt


# 噪声添加函数
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

    salt_coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255

    pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0

    return noisy_image


# 滤波操作和展示
def filter_and_display(noisy_img, title, filter_func):
    filtered_img = filter_func(noisy_img)

    # 绘图
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(noisy_img, cmap='gray')
    axs[0].set_title(f'{title} (Noisy)')
    axs[0].axis('off')

    axs[1].imshow(filtered_img, cmap='gray')
    axs[1].set_title(f'{title} (Filtered)')
    axs[1].axis('off')

    plt.show()


# 读取图像并添加噪声
image = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

gaussian_noise_img = add_gaussian_noise(image)
uniform_noise_img = add_uniform_noise(image)
salt_pepper_noise_img = add_salt_pepper_noise(image)


# 1. 高斯噪声去噪 - 使用高斯滤波器
def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 1)


filter_and_display(gaussian_noise_img, "Gaussian Noise", gaussian_filter)


# 2. 均匀噪声去噪 - 使用均值滤波器
def mean_filter(image):
    return cv2.blur(image, (5, 5))


filter_and_display(uniform_noise_img, "Uniform Noise", mean_filter)


# 3. 椒盐噪声去噪 - 使用中值滤波器
def median_filter(image):
    return cv2.medianBlur(image, 5)


filter_and_display(salt_pepper_noise_img, "Salt & Pepper Noise", median_filter)

