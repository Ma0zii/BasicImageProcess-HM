import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def plt_imshow(m):
    # m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    plt.imshow(m)
    plt.axis("off")
    plt.show()

def maxPixel(m):
    m = np.array(m)
    a = np.max(m)
    return a

def darkChannel():
    for i in range(rows):
        for j in range(cols):
            minDt = min(img[i][j][0], img[i][j][1], img[i][j][2])
            for k in range(channels):
                dc[i][j][k] = minDt

kenlRatio = 0.01
minAtomsLight = 240

img_name = "image_work2/9.jpg"
img = cv2.imread(img_name) # 读取图片
cv2.imshow("original", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_arr = np.array(img)

rows, cols, channels = img.shape # 行，列，通道数
dc = img

darkChannel() # 暗通道

krnlsz = max(3, rows * kenlRatio, cols * kenlRatio)  # 滤波窗口尺寸
dc2 = cv2.erode(dc, np.ones((math.ceil(krnlsz), math.ceil(krnlsz)))) # 最小值滤波
plt_imshow(dc2)

'''归一化'''
t = 255 - dc2
t_d = np.array(t) / 255
t_d = sum(sum(t_d)/(rows * cols))
t_d = sum(t_d) / 3
print(t_d)

d = np.array(dc2)
A = min(minAtomsLight, maxPixel(d))

J = np.empty([rows, cols, channels], dtype=int)
for i in range(3):
    J[:, :, i] = (img_arr[:, :, i] - (1 - t_d) * A) / t_d
plt_imshow(J)






