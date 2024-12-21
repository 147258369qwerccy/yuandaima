import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_prewitt_operator(image):
    # Prewitt算子的水平和垂直方向卷积核
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)

    # 应用卷积核计算梯度，并将结果转换为float32类型
    grad_x = cv2.filter2D(image, -1, kernelx).astype(np.float32)
    grad_y = cv2.filter2D(image, -1, kernely).astype(np.float32)

    return grad_x, grad_y


# 读取图像
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("Error: Unable to load image.")
else:
    # 应用Prewitt梯度算子
    grad_x, grad_y = apply_prewitt_operator(image)

    # 计算梯度幅值
    magnitude = cv2.magnitude(grad_x, grad_y)

    # 显示原始图像和梯度图像
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Gradient X')
    plt.imshow(grad_x, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Gradient Magnitude')
    plt.imshow(magnitude, cmap='gray')
    # 如果你也想显示Grad_Y，可以添加另一个subplot
    # 但为了简洁，这里只显示梯度幅值

    plt.show()
