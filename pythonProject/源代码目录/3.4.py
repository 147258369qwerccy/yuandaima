import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义Kirsch算子卷积核
kernels = [
    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # 0度方向
    np.array([[-3, -3, -3], [-3, 0, 5], [5, 5, 5]]),  # 45度方向
    np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # 90度方向
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # 135度方向
    np.array([[-3, -3, -3], [5, 0, -3], [5, 5, 5]]),  # 180度方向
    np.array([[-3, -3, 5], [-3, 0, -3], [-3, 5, 5]]),  # 225度方向
    np.array([[-3, -3, 5], [-3, 0, -3], [5, 5, -3]]),  # 270度方向
    np.array([[-3, 5, 5], [-3, 0, -3], [-3, -3, -3]])  # 315度方向
]


def kirsch_edge_detection(image):
    # 读取图像并转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 初始化输出图像，存储每个像素的最大响应
    edge_image = np.zeros_like(gray_image, dtype=np.float32)

    # 对每个方向进行卷积操作
    for kernel in kernels:
        # 卷积操作，得到该方向的梯度
        grad = cv2.filter2D(gray_image, -1, kernel)

        # 取每个位置的最大值，表示边缘强度
        edge_image = np.maximum(edge_image, grad)

    # 将结果转换为0-255的整数类型（边缘图像）
    edge_image = np.uint8(np.abs(edge_image))

    # 可以进行二值化操作来强调边缘
    _, edge_image = cv2.threshold(edge_image, 100, 255, cv2.THRESH_BINARY)

    return edge_image


# 加载输入图像
image = cv2.imread('tuxi4.jpg')  # 替换为你的图像文件路径

# 应用Kirsch边缘检测
edges = kirsch_edge_detection(image)

# 显示原图和边缘图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Kirsch Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
