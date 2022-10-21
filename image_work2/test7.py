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

file = open(r"image_work2/1.txt", "a", encoding='utf-8')
img_name = "image_work2/9.jpg"
img = cv2.imread(img_name) # 读取图片
cv2.imshow("original", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转化通道


file.write(str(np.array(img)))
file.write("\n")

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


file.write(str(t))
file.write("\n")
file.close()

print(t)
d = np.array(img_mf)
A = min(minAtomsLight, maxPixel(d))
print(A)

J = np.empty([rows, cols, channels], dtype=int)
for i in range(3):
    # J[:, :, i] = (img_arr[:, :, i] + (t[:, :, i] - 1) * A) / t[:, :, i]
    # J[:, :, i] = (img_arr[:, :, i] - t[:, :, i]) / (1 - t[:, :, i] / A)
    J[:, :, i] = (img_arr[:, :, i] + (t - 1) * A) / t


# J = cv2.cvtColor(J, cv2.COLOR_RGB2BGR) # 转化通道
plt_imshow(J)






