import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness

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

# 导向滤波
def guideFilter(I, p, winSize, eps):
    r = winSize
    m_I = cv2.boxFilter(I, -1, (r, r))# I的均值平滑
    m_p = cv2.boxFilter(p, -1, (r, r))# p的均值平滑
    m_II = cv2.boxFilter(I * I, -1, (r, r))# I*I的均值平滑
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))# I*p的均值平滑

    var_I = m_II - m_I * m_I# 方差
    cov_Ip = m_Ip - m_I * m_p# 协方差

    a = cov_Ip / (var_I + eps)# 相关因子a
    b = m_p - a * m_I# 相关因子b

    m_a = cv2.boxFilter(a, -1, (r, r))# 对a进行均值平滑
    m_b = cv2.boxFilter(b, -1, (r, r))# 对b进行均值平滑

    q = m_a * I + m_b
    return q

t0 = 0.1
winRatio = 0.01
minAtomsLight = 240

img_name = "image_work2/7.jpg"
img = cv2.imread(img_name) # 读取图片
cv2.imshow("original", img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转化通道

img_arr = np.array(img)

rows, cols, channels = img.shape # 行，列，通道数
winSize = math.ceil(max(3, rows * winRatio, cols * winRatio)) # 滤波窗口尺寸

img_dc = darkChannel(img, rows, cols, channels)
# plt_imshow(img_dc)
img_mf = minFilter(img_dc, winSize)
# plt_imshow(img_mf)
img_gf = guideFilter(img_dc / 255.0, img_mf / 255.0, winSize * 4, eps=0.001)
plt_imshow(img_gf)

'''归一化'''
t = np.array(img_gf)
t = sum(sum(t)/(rows * cols))
t = sum(t) / 3

d = np.array(img_mf)
A = min(minAtomsLight, maxPixel(d))


J = np.empty([rows, cols, channels], dtype=int)
for i in range(3):
    # J[:, :, i] = (img_arr[:, :, i] + (t[:, :, i] - 1) * A) / t[:, :, i]
    # J[:, :, i] = (img_arr[:, :, i] - t[:, :, i]) / (1 - t[:, :, i] / A)
    J[:, :, i] = (img_arr[:, :, i] + (t - 1) * A) / t

cv2.imwrite('image_work2/out.jpg', J)
jl = cv2.imread('image_work2/out.jpg')
jl = aug(jl)
cv2.imwrite('image_work2/out.jpg', jl)





