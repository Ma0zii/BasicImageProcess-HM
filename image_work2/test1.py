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

# 暗通道
def darkChannel(m, rows, cols, channels):
    dc = m
    for i in range(rows):
        for j in range(cols):
            minDt = min(m[i][j][0], m[i][j][1], m[i][j][2])
            for k in range(channels):
                dc[i][j][k] = minDt
    return dc

# 最小值滤波
def minFilter(m, winSize):
    mf = cv2.erode(m, np.ones((winSize, winSize))) # 腐蚀
    return mf

t0 = 0.1
winRatio = 0.01
minAtomsLight = 240

img = cv2.imread("image_work2/9.jpg") # 读取图片
cv2.imshow("original", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转化通道
img_arr = np.array(img)

rows, cols, channels = img.shape # 行，列，通道数
winSize = math.ceil(max(3, rows * winRatio, cols * winRatio)) # 滤波窗口尺寸

img_dc = darkChannel(img, rows, cols, channels)
img_mf = minFilter(img_dc, winSize)
plt_imshow(img_mf)

t = np.array(img_mf)
t = sum(sum(sum(t) / (rows * cols))) / 3

d = np.array(img_mf)
A = min(minAtomsLight, maxPixel(d))

J = np.empty([rows, cols, channels], dtype=int)
for i in range(3):
    J[:, :, i] = (img_arr[:, :, i] + (t - 1) * A) / t
plt_imshow(J)