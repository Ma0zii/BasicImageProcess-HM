from PIL import Image
import numpy as np
import cv2

# image_work1/image_test.jpeg

def testChangeRGB():
    a = np.array(Image.open('image_work1/image_test.jpeg'))
    imgold = cv2.imread("image_work1/image_test.jpeg")
    # b = [255, 255, 255] - a
    b = [255, 255, 255] - a
    im = Image.fromarray(b.astype('uint8'))
    im.save('image_work1/image_new.jpg')
    img = cv2.imread("image_work1/image_new.jpg")
    # 获取输出的检测图片
    cv2.imshow('ImgOld', imgold)
    cv2.imshow('SmileNew', img)
    c = cv2.waitKey(0)  # 按任意键继续

##图片的缩放
def testScal():
    # -*- coding: UTF-8 -*-
    # 读取图像
    im = Image.open("image_work1/image_test.jpeg")
    im.show()
    # 原图像缩放为128x128
    im_resized = im.resize((128, 128))
    im_resized.show()

testChangeRGB()