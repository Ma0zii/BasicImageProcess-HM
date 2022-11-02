import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像信息
from numpy.fft import ifftshift

img0 = cv2.imread("1.jpg")
img1 = cv2.resize(img0, dsize=None, fx=1, fy=1)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
h, w = img1.shape[:2]
print(h, w)
# cv2.namedWindow("W0")
# cv2.imshow("W0", img2)
# cv2.waitKey(delay=0)
# 将图像转化到频域内并绘制频谱图
##numpy实现
plt.rcParams['font.family'] = 'SimHei'  # 将全局中文字体改为黑体
f = np.fft.fft2(img2)
fshift = np.fft.fftshift(f)  # 将0频率分量移动到图像的中心
magnitude_spectrum0 = 20 * np.log(np.abs(fshift))
# 傅里叶逆变换
# # Numpy实现
# ifshift = np.fft.ifftshift(fshift)
# # 将复数转为浮点数进行傅里叶频谱图显示
# ifimg = np.log(np.abs(ifshift))
# if_img = np.fft.ifft2(ifshift)
# origin_img = np.abs(if_img)
# imggroup = [img2, magnitude_spectrum0, ifimg, origin_img]
# titles0 = ['原始图像', '经过移动后的频谱图', '逆变换得到的频谱图', '逆变换得到的原图']
# for i in range(4):
#     plt.subplot(2, 2, i + 1)
#     plt.xticks([])  # 除去刻度线
#     plt.yticks([])
#     plt.title(titles0[i])
#     plt.imshow(imggroup[i], cmap='gray')
# plt.show()
##OpenCV实现
dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.subplot(121), plt.imshow(img2, cmap='gray')
plt.title('原图'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum1, cmap='gray')
plt.title('频谱图'), plt.xticks([]), plt.yticks([])
plt.show()
