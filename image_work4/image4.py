import cv2
import math
import numpy as np

# 移动
def move(img):
    height, width, channels = img.shape
    emptyImage2 = img.copy()
    x = 20
    y = 20
    for i in range(height):
        for j in range(width):
            if i >= x and j >= y:
                emptyImage2[i, j] = img[i - x][j - y]
            else:
                emptyImage2[i, j] = (0, 0, 0)
    return emptyImage2

# 水平镜像
def mirror(img):
    h, w, c = img.shape
    newImage = img.copy()
    for i in range(h):
        a = 0
        for j in range(w - 1, 0, -1):
            for k in range(c):
                newImage[i][a][k] = img[i][j][k]
            a += 1
    return newImage

img = cv2.imread("1.jpg")

SaltImage = move(img)
mirrorimg = mirror(img)

cv2.imshow("Image", img)
cv2.imshow("mm", mirrorimg)
cv2.waitKey(0)


# cv2.imshow("Image", img)
# cv2.imshow("ss", SaltImage)
# cv2.waitKey(0)


