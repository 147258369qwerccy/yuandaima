import cv2
import numpy as np

def convert_bgr_to_hsi(bgr_image):

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)
    i = v  # 在这个简化的例子中，我们使用V作为I


    hsi_image = cv2.merge((h, s, i)).astype(np.uint8)  # 注意：这里我们可能需要类型转换，因为h可能是float类型


    return hsi_image, i  # 返回近似的HSI图像和I分量（用于后续均衡化）


def histogram_equalization_hsi(bgr_image):
    # 将BGR图像转换为近似的HSI图像，并提取I分量
    hsi_image, i_channel = convert_bgr_to_hsi(bgr_image)

    # 对I分量进行直方图均衡化
    i_equalized = cv2.equalizeHist(i_channel)

    hsi_image[..., 2] = i_equalized  # 假设I分量是hsi_image的第三个通道（对应于V在HSV中）

    # 正确的做法：显示或处理均衡化后的I分量（灰度图像）
    gray_image = i_equalized  # 这实际上是均衡化后的I分量（或原始的V分量）

    return gray_image  # 返回均衡化后的I分量作为灰度图像


# 读取图像
bgr_image = cv2.imread('tuxi4.jpg')

# 进行HSI空间上的直方图均衡化（实际上只处理了近似的I分量）
equalized_i_channel = histogram_equalization_hsi(bgr_image)

# 显示结果（以灰度图像的形式显示均衡化后的I分量）
cv2.imshow('Equalized Intensity Channel', equalized_i_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

