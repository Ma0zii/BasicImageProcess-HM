import cv2 as cv
import numpy as np


def watershed_demo():
   # print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100) #边缘保留滤波去噪
    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY) #转化为灰度图
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) #二值化
    cv.imshow("binary-image", binary)

    # morphology operation形态学操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) #构造结构
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2) #连续两次开操作（去除图像中的任何小的白噪声）；闭运算（为了去除物体上的小洞）
    sure_bg = cv.dilate(mb, kernel, iterations=3)  #连续三次膨胀操作
    cv.imshow("mor-opt", sure_bg)

    # distance transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)#距离变化（提取出我们确信它们是硬币的区域）
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)#归一化
    dist_output1 = np.uint8(dist_output)


    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)

    surface_fg = np.uint8(surface)#将float类型转化为uint
    cv.imshow("surface-bin", surface_fg)
    unknown = cv.subtract(sure_bg, surface_fg)#除种子以外的区域（剩下的区域是我们不知道的区域，无论是硬币还是背景.分水岭算法应该找到它）
    ret, markers = cv.connectedComponents(surface_fg)
    #求连通区域（创建标记：它是一个与原始图像相同大小的数组，但使用int32数据类型，并对其内部的区域进行标记.）
    # watershed transform 分水岭变换
    markers = markers + 1 # Add one to all labels so that sure background is not 0, but 1
    markers[unknown==255] = 0 #  mark the region of unknown with zero
    markers = cv.watershed(src, markers=markers)
    src[markers==-1] = [0, 0, 255]#标记
    cv.imshow("result", src)

src = cv.imread("dog1.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
watershed_demo()
cv.waitKey(0)

cv.destroyAllWindows()
