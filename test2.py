import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def spilt(a):
  if a % 2 == 0:
    x1 = x2 = a / 2
  else:
    x1 = math.floor(a / 2)
    x2 = a - x1
  return -x1, x2

def original(i, j, k, a, b, img):
  x1, x2 = spilt(a)
  y1, y2 = spilt(b)
  temp = np.zeros(a * b)
  count = 0
  for m in range(x1, x2):
    for n in range(y1, y2):
      if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
        temp[count] = img[i, j, k]
      else:
        temp[count] = img[i + m, j + n, k]
      count += 1
  return temp

def min_functin(a, b, img):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                img[i, j, k] = np.min(temp)
    return img


def darkChannel():
    for i in range(rows):
        for j in range(cols):
            minDt = min(img_ary[i][j][0], img_ary[i][j][1], img_ary[i][j][2])
            for k in range(channels):
                dc[i][j][k] = minDt

kenlRatio = 0.01
minAtomsLight = 240

img_name = "image_work2/1.jpg"
img = cv.imread(img_name) # 读取图片
cv.imshow("original", img)

img_ary = np.array(img)
rows, cols, channels = img.shape # 行，列，通道数
dc = np.empty([rows, cols, channels], dtype=int)


darkChannel() # 暗通道
plt.imshow(dc)
plt.axis("off")
plt.show()

krnlsz = max(3, rows * kenlRatio, cols * kenlRatio)  # 滤波窗口尺寸
# dc2 = cv.erode(dc, np.ones((10, 10)))
dc2 = min_functin(math.ceil(krnlsz), math.ceil(krnlsz), copy.copy(dc))

plt.imshow(dc2)
plt.axis("off")
plt.show()


# kernel=np.ones((7, 7), dtype=np.uint8)
# abc = cv.erode(src = dc, kernel = kernel, iterations=1)
# plt.imshow(abc)
# plt.axis("off")
# plt.show()






