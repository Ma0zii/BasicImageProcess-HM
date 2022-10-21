import cv2
import numpy as np
from matplotlib import pyplot as plt

#matplotlib标题字体设置
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

#调整图像大小
plt.figure(figsize=(20, 20))

#读取图像
img = cv2.imread('1.jpg', 0)  # 直接读为灰度图像
# numpy傅里叶变换
f = np.fft.fft2(img)
# 移位
f_shift = np.fft.fftshift(f)

# 取绝对值：将复数变化成实数
# 取对数的目的为了将数据变化到较小的范围（比方0-255）
s1 = np.log(np.abs(f))
s2 = np.log(np.abs(f_shift))

# 进行傅里叶逆变换 将频率谱转为图像
img_back = np.fft.ifft2(f)
img_back = np.abs(img_back)

# 移位到中点的频率图
plt.subplot(121), plt.imshow(s2, 'gray'), plt.title('频域图')
#展示图像
plt.subplot(122), plt.imshow(img_back, 'gray'), plt.title('时域图')
plt.show()