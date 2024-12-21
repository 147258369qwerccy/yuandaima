import cv2
from matplotlib import pyplot as plt

# 加载Haar Cascade模型用于人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
img = cv2.imread('tuxi4.jpg')

# 将图像转换为灰度图像（Haar Cascade只在灰度图像上进行处理）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Haar Cascade进行人脸检测
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 在检测到的人脸上绘制矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 使用matplotlib显示图像
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换为RGB以便显示
plt.axis('off')
plt.title('Face Detection using Haar Cascade')
plt.show()
