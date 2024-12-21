import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. 空间域拉普拉斯变换
def laplacian_spatial(image):

    # 使用cv2.Laplacian函数计算图像的拉普拉斯算子，数据类型为cv2.CV_64F（64位浮点数）
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # 将计算得到的拉普拉斯算子结果转换为绝对值（处理可能的负数情况），并转换为uint8类型（便于显示等操作）
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    return laplacian_abs


# 2. 频率域拉普拉斯变换
def laplacian_frequency(image):

    # 将图像转换为float32类型，这是进行傅里叶变换要求的数据类型
    img_float = np.float32(image)
    # 进行二维离散傅里叶变换（DFT）
    dft = np.fft.fft2(img_float)
    # 将零频率分量移动到频谱中心，方便后续处理和可视化
    dft_shifted = np.fft.fftshift(dft)

    # 获取图像的行数和列数
    rows, cols = image.shape
    # 计算频谱中心坐标
    crow, ccol = rows // 2, cols // 2

    # 创建拉普拉斯滤波器的频域掩码（一个与图像尺寸相同的矩阵）
    laplacian_kernel = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # 计算每个位置到频谱中心的距离
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            # 根据拉普拉斯滤波器在频域的计算公式设置对应的值
            laplacian_kernel[i, j] = -4 * np.pi ** 2 * distance ** 2

    # 在频域对图像应用拉普拉斯滤波器，通过对应元素相乘
    filtered_dft = dft_shifted * laplacian_kernel
    # 将零频率分量移回原来的位置
    filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
    # 进行逆傅里叶变换，恢复图像
    img_reconstructed = np.fft.ifft2(filtered_dft_shifted)
    # 取逆傅里叶变换结果的绝对值（幅度），并转换为uint8类型便于显示
    img_reconstructed = np.abs(img_reconstructed).astype(np.uint8)

    return img_reconstructed


# 3. 主函数，用于读取图像、调用上述函数进行处理并展示结果
def main():
    # 读取图像（这里以彩色图像为例，后续会转换为灰度图），你需要将'your_image.jpg'替换为实际的图像路径
    image = cv2.imread('tuxi4.jpg')
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 空间域拉普拉斯锐化
    spatial_result = laplacian_spatial(gray_image)
    # 频率域拉普拉斯锐化
    frequency_result = laplacian_frequency(gray_image)

    # 展示原始图像
    plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('Original Image')
    plt.axis('off')
    # 展示空间域拉普拉斯锐化后的图像
    plt.subplot(132), plt.imshow(spatial_result, cmap='gray'), plt.title('Spatial Laplacian')
    plt.axis('off')
    # 展示频率域拉普拉斯锐化后的图像
    plt.subplot(133), plt.imshow(frequency_result, cmap='gray'), plt.title('Frequency Laplacian')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()




