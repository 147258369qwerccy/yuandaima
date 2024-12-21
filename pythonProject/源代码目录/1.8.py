import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = 'tuxi4.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("Error: Unable to load image.")
else:
    # 定义拉普拉斯算子
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)

    # 应用拉普拉斯算子进行锐化
    sharpened = cv2.filter2D(image, -1, laplacian)


    sharpened_abs = np.absolute(sharpened)
    sharpened_8u = np.uint8(sharpened_abs)

    # 显示原始图像和锐化后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Sharpened Image using Laplacian')
    plt.imshow(sharpened_8u, cmap='gray')

    plt.show()
