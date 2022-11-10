import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def OTSU(img_array):            #传入的参数为ndarray形式
    height = img_array.shape[0]
    width = img_array.shape[1]
    count_pixel = np.zeros(256)

    for i in range(height):
        for j in range(width):
            count_pixel[int(img_array[i][j])] += 1

    fig = plt.figure()        #通过绘制直方图可以观察像素的分布情况
    ax = fig.add_subplot(111)
    ax.bar(np.linspace(0, 255, 256), count_pixel)
    ax.set_xlabel("pixels")
    ax.set_ylabel("num")
    plt.show()

    max_variance = 0.0
    best_thresold = 0
    for thresold in range(256):
        n0 = count_pixel[:thresold].sum()
        n1 = count_pixel[thresold:].sum()
        w0 = n0 / (height * width)
        w1 = n1 / (height * width)
        u0 = 0.0
        u1 = 0.0

        for i in range(thresold):
            u0 += i * count_pixel[i]
        for j in range(thresold, 256):
            u1 += j * count_pixel[j]

        u = u0 * w0 + u1 * w1
        tmp_var = w0 * np.power((u - u0), 2) + w1 * np.power((u - u1), 2)

        if tmp_var > max_variance:
            best_thresold = thresold
            max_variance = tmp_var

    return best_thresold
img = cv2.imread("dog1.jpg", 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.resize(img, (200, 200))#大小
# img = np.array(img).astype(np.float32)
best_thresold = OTSU(img)
ret2, th2 = cv2.threshold(img, best_thresold, 255, cv2.THRESH_BINARY)
print(ret2)
plt.imshow(th2, cmap=cm.gray)
plt.show()