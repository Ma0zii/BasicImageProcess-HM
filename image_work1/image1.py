from PIL import Image
import numpy as np
import cv2

# image_work1/image_test.jpeg

def testChangeRGB():
    a = np.array(Image.open('image_test.jpeg'))
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
    im = Image.open("image_test.jpeg")
    im.show()
    # 原图像缩放为128x128
    im_resized = im.resize((128, 128))
    im_resized.show()

##图片的旋转
def testRotate():
    # -*- coding: UTF-8 -*-
    from PIL import Image
    # 读取图像
    im = Image.open("image_test.jpeg")
    im.show()
    # 指定逆时针旋转的角度
    im_rotate = im.rotate(45)
    im_rotate.show()

def testTranspose():
    # 读取图像
    im = Image.open("image_test.jpeg")
    out = im.transpose(Image.FLIP_LEFT_RIGHT)
    out = im.transpose(Image.FLIP_TOP_BOTTOM)
    out = im.transpose(Image.ROTATE_90)
    out = im.transpose(Image.ROTATE_180)
    out = im.transpose(Image.ROTATE_270)
    out.show()


# 图像的手绘
def testRonaldo():
    from PIL import Image
    import numpy as np

    a = np.asarray(Image.open('image_test.jpeg').convert('L')).astype('float')

    depth = 10.  # (0-100)
    grad = np.gradient(a)  # 取图像灰度的梯度值
    grad_x, grad_y = grad  # 分别取横纵图像梯度值
    grad_x = grad_x * depth / 100.
    grad_y = grad_y * depth / 100.
    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A

    vec_el = np.pi / 2.2  # 光源的俯视角度，弧度值
    vec_az = np.pi / 4.  # 光源的方位角度，弧度值
    dx = np.cos(vec_el) * np.cos(vec_az)  # 光源对x 轴的影响
    dy = np.cos(vec_el) * np.sin(vec_az)  # 光源对y 轴的影响
    dz = np.sin(vec_el)  # 光源对z 轴的影响

    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # 光源归一化
    b = b.clip(0, 255)

    imgold = cv2.imread("image_work1/image_test.jpeg")
    cv2.imshow('RonaldoOld', imgold)
    im = Image.fromarray(b.astype('uint8'))  # 重构图像
    im.save('image_work1/Ronaldo.jpg')
    img = cv2.imread("image_work1/Ronaldo.jpg")
    # 获取输出的检测图片
    cv2.imshow('RonaldoNew', img)
    c = cv2.waitKey(0)



# testChangeRGB()
# testScal()
# testRotate()
# testTranspose()
testRonaldo()