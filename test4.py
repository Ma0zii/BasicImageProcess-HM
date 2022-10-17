import copy

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def plt_imshow(m):
    plt.imshow(m)
    plt.axis("off")
    plt.show()

def maxP(m):
    a = -1
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            for k in range(m.shape[2]):
                if m[i][j][k] > a:
                    a = m[i][j][k]
    return a


def darkChannel():
    for i in range(rows):
        for j in range(cols):
            minDt = min(img[i][j][0], img[i][j][1], img[i][j][2])
            for k in range(channels):
                dc[i][j][k] = minDt

kenlRatio = 0.01
minAtomsLight = 240

img_name = "image_work2/2.jpg"
img = cv.imread(img_name) # 读取图片
cv.imshow("original", img)

rows, cols, channels = img.shape # 行，列，通道数
dc = img

darkChannel() # 暗通道
plt_imshow(dc)

krnlsz = max(3, rows * kenlRatio, cols * kenlRatio)  # 滤波窗口尺寸
dc2 = cv.erode(dc, np.ones((math.ceil(krnlsz), math.ceil(krnlsz)))) # 最小值滤波
plt_imshow(dc2)

'''归一化'''
t = 255 - dc2
t_d = np.array(t)
t_d = sum(sum(t_d)/(rows * cols))
t_d = sum(t_d) / 3

d = np.array(dc2)
A = min(minAtomsLight, maxP(d))

J = np.empty([rows, cols, channels], dtype=int)
plt_imshow(img)
for i in range(3):
    J[:, :, i] = (img[:, :, i] - (1 - t_d) * A) / t_d
plt_imshow(J)






