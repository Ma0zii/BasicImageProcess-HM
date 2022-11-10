import numpy as np
import math
import cv2

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1/255.0 - img2/255.0) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img = cv2.imread('6.jpg')
img_8bit = cv2.imread('1_8.jpg')
img_4bit = cv2.imread('1_4.jpg')
img_2bit = cv2.imread('1_2.jpg')

psnr_8 = psnr(img, img_8bit)
psnr_4 = psnr(img, img_4bit)
psnr_2 = psnr(img, img_2bit)

print("psnr的值为：")
print(psnr_8)
print(psnr_4)
print(psnr_2)