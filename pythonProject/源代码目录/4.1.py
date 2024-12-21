import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取图像，灰度图
img = cv2.imread('tuxi4.jpg', cv2.IMREAD_GRAYSCALE)

# 显示原始图像
plt.figure(figsize=(6, 6))
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# 将图像展平为二维数组，每列作为一个样本
# 这里我们将图像的每一列作为一个样本（如果你有多张图像，每张图像就成为一个样本）
img_flatten = img.T  # 转置，使得每列是一个特征

# 创建PCA对象，设置主成分的数量
n_components = 50  # 提取50个主成分
pca = PCA(n_components=n_components)

# 执行PCA降维
img_pca = pca.fit_transform(img_flatten)  # 每列像素作为样本

# 将PCA的逆变换结果恢复为原始空间
img_reconstructed = pca.inverse_transform(img_pca)

# 恢复为二维图像
img_reconstructed = img_reconstructed.T

# 显示重建后的图像
plt.figure(figsize=(6, 6))
plt.title(f'Reconstructed Image with {n_components} Components')
plt.imshow(img_reconstructed, cmap='gray')
plt.axis('off')
plt.show()

# 显示PCA的主成分（前几个主成分）
for i in range(n_components):
    plt.figure(figsize=(6, 6))
    plt.title(f'Principal Component {i + 1}')
    plt.imshow(pca.components_[i].reshape(img.shape), cmap='gray')
    plt.axis('off')
    plt.show()

# 输出方差解释率，查看每个主成分解释的方差比例
print("Explained variance ratio:", pca.explained_variance_ratio_)

