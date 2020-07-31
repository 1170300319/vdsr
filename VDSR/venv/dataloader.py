from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision as ptv
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import utils
import h5py
import scipy.io as sio
import eval


class dataset(Dataset):
    def __init__(self, xss, yss):
        self.data = xss
        self.target = yss

    def __getitem__(self, item):
        return torch.from_numpy(self.data[item, :, :, :]).float(), torch.from_numpy(self.target[item, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]


# 原始图片处理成高清图片
def preprocess_img():
    roots = [i for i in range(50)]
    for root in roots:
        i = str(root + 1) if root > 8 else "0" + str(root + 1)
        img = Image.open("imgs/" + i + ".jpg")
        w, h = img.size  # 大小
        n = w / 512  # 缩放系数
        img.thumbnail((w / n, h / n))
        w_, h_ = img.size
        print("old " + str((w, h)) + " new " + str((w_, h_)))
        img.save("imgs_512/" + i + ".jpg")


# 高清图片处理成低清图片
def half_img():
    roots = [i for i in range(50)]
    for root in roots:
        i = str(root + 1) if root > 8 else "0" + str(root + 1)
        img = Image.open("imgs_512/" + i + ".jpg")
        w, h = img.size  # 大小
        n = w / 256  # 缩放系数
        img.thumbnail((w / n, h / n))
        w_, h_ = img.size
        print("old " + str((w, h)) + " new " + str((w_, h_)))
        img.save("imgs_256/" + i + ".jpg")


# 插值放大低清图像
def boarder_img_to_512(img, name):
    h, w = img.shape
    n = 512/w
    img = Image.fromarray(img)
    bmg = img.resize((512, int(h*n)))
    bmg.save("imgs_256to512/"+name)
    #retimg = utils.bicubic_interpolation(img, dstW=512, dstH=int(h*n))
    #retimg = Image.fromarray(retimg)
    #retimg.save("imgs_256to512/"+name)


# 获取大图像
def get_xy(show_img=False):
    roots = [i for i in range(50)]
    xs, ys = [], []
    for root in roots:
        i = str(root + 1) if root > 8 else "0" + str(root + 1)
        x = Image.open("imgs_256to512/" + i + ".jpg").convert("L")
        y = Image.open("imgs_512/" + i + ".jpg").convert("L")
        #xs.append(np.array(x)/255.)
        #ys.append(np.array(y)/255.)
        xs.append(np.array(x))
        ys.append(np.array(y))
        if show_img:
            plt.subplot(2, 1, 1)
            plt.title("x")
            plt.imshow(x)
            plt.subplot(2, 1, 2)
            plt.title("y")
            plt.imshow(y)
            plt.show()
    # xx = torch.utils.utils.data.DataLoader(train, batch_size=batchSize)
    return xs, ys


# 加载dataset
def loadh5():
    xs, ys = get_xy()
    xss, yss = split_img(xs), split_img(ys)
    xss, yss = my_agrumentation(xss, yss)
    h5 = dataset(xss, yss)
    #hf = h5py.File("train.h5")
    #xs, ys = hf.get("data"), hf.get("label")
    #h5 = dataset(xs, ys)
    return h5

# 数据增强
def my_agrumentation(xs, ys):
    xss, yss = [], []
    for i in range(len(xs)):
        xss.append(xs[i])
        yss.append(ys[i])
        x = Image.fromarray(xs[i][0])
        y = Image.fromarray(ys[i][0])
        x_arr = np.array([np.array(tfs.RandomHorizontalFlip(p=1)(x))])  # 水平
        xss.append(x_arr)
        y_arr = np.array([np.array(tfs.RandomHorizontalFlip(p=1)(y))])
        yss.append(y_arr)
        x_arr = np.array([np.array(tfs.RandomVerticalFlip(p=1)(x))])  # 垂直
        xss.append(x_arr)
        y_arr = np.array([np.array(tfs.RandomVerticalFlip(p=1)(y))])
        yss.append(y_arr)
        x_arr = np.array([np.array(tfs.RandomRotation(45)(x))])  # 旋转
        xss.append(x_arr)
        y_arr = np.array([np.array(tfs.RandomRotation(45)(y))])
        yss.append(y_arr)
    return np.array(xss), np.array(yss)


# 图像裁剪
def split_img(xs):
    xss = []  # 裁剪后的图像
    for i in range(len(xs)):
        x = xs[i]
        for j in range(int(x.shape[0] / 41)):
            for k in range(int(x.shape[1] / 41)):
                xss.append([x[41 * j:41 * (j + 1), 41 * k:41 * (k + 1)]])
    return np.array(xss)


if __name__ == '__main__':
    # preprocess_img()
    # half_img()
    '''
    net = torch.load("checkpoint/model_epoch_90.pth", map_location=lambda storage, loc: storage)["model"]
    net = net.cuda()

    h5 = loadh5()
    data = h5.data
    target = h5.target
    #print(data, target)
    im_gt_y = sio.loadmat("baby_GT_x2.mat")["im_gt_y"]
    im_b_y = sio.loadmat("baby_GT_x2.mat")["im_b_y"]
    print(eval.PSNR(im_b_y, im_gt_y))

    im_input = im_b_y/255.0
    print(im_input.shape[0], im_input.shape[1])
    im_input = torch.autograd.Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    im_input = im_input.cuda()
    HR = net(im_input)
    im_h = HR.cpu().data[0].numpy().astype(np.float)
    im_h = im_h*255.0
    im_h[im_h<0]=0
    im_h[im_h>255.0]=255.0
    im_h = im_h[0,:,:]
    print(eval.PSNR(im_gt_y, im_h))
    
    '''

    #xs, ys = get_xy()
    #xss, yss = split_img(xs), split_img(ys)
    #print(xss.shape, yss.shape)
    roots = [i for i in range(19)]
    xs, ys = [], []
    for root in roots:
        i = str(root + 1) if root > 8 else "0" + str(root + 1)
        x = Image.open("数据集/" + i + ".jpg").convert("L")
        boarder_img_to_512(np.array(x), name=i+".jpg")