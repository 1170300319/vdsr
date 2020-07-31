from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt


# 产生16个像素点的权重
def bibuic(x):
    x = abs(x)
    if x <= 1:
        return 1-2*(x**2)+x**3
    elif x < 2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0


# 双三次插值
def bicubic_interpolation(img, dstW, dstH):
    srcH, srcW = img.shape
    retimg = np.zeros((dstH, dstW), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            srcx = i*(srcH/dstH)
            srcy = j*(srcW/dstW)
            x = math.floor(srcx)
            y = math.floor(srcy)
            u = srcx-x
            v = srcy-y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or j + jj < 0 or x + ii >= srcH or y + jj >= srcW:
                        continue
                    tmp += img[x+ii, y+jj]*bibuic(ii-u)*bibuic(jj-v)
            retimg[i, j] = np.clip(tmp, 0, 255)
    return retimg


if __name__ == '__main__':
    LR = Image.open("imgs_256/01.jpg").convert("L")
    HR = Image.open("imgs_512/01.jpg").convert("L")
    w, h = HR.size
    LR_arr = np.array(LR)
    LR_arr_ = bicubic_interpolation(LR_arr, dstW=w, dstH=h)
    plt.subplot(2, 2, 1)
    plt.imshow(LR, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(Image.fromarray(LR_arr_))
    plt.subplot(2, 2, 3)
    plt.imshow(HR, cmap='gray')
    plt.show()