import torch
from PIL import Image
import matplotlib.pyplot as plt
import dataloader
import numpy as np
import math


def PSNR(pred, gt):
    imdff = pred-gt
    #print("imdff"+str(imdff))
    rmse = math.sqrt(np.mean(imdff**2))
    #print("rmse"+str(rmse))
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)


if __name__ == '__main__':
    net = torch.load("checkpoint/model_epoch_140.pth", map_location=lambda storage, loc: storage)["model"]
    net = net.cuda()
    roots = ['0'+str(i+1) for i in range(9)]
    psnrSum = 0
    for eachRoot in roots:
        test_img = Image.open("imgs_256to512/"+eachRoot+".jpg")
        test_img = test_img.convert("L")
        ori_img = Image.open("imgs_512/"+eachRoot+".jpg")
        ori_img = ori_img.convert("L")
        train_x, train_y = dataloader.get_xy()
        train_x = dataloader.split_img(train_x)
        train_y = dataloader.split_img(train_y)
        #arr = np.array(test_img)/255.0  # 归一
        arr = np.array(test_img)  # 归一
        ori_arr = np.array(ori_img)
        xs = dataloader.split_img([arr])
        #print(train_x[0, 0])
        #print(train_y[0, 0])
        #print(xs[0, 0])
        HRs = []
        for each in xs:
            each = torch.tensor(each).float().cuda()
            each = torch.autograd.Variable(each).view(-1, 1, each.shape[0], each.shape[1])
            HRs.append(net(each))

        HRss = []
        for each in HRs:
            each = each.cpu().detach().numpy()
            each = each.reshape(1, 1, 41, 41)
            #HRss.append(each[0][0]*255.0)  # 复原
            HRss.append(each[0][0])  # 复原
            #print(each[0][0])
        #print(HRss[0]/255.)
        new_img = np.zeros(arr.shape)
        w, h = int(arr.shape[0] / 41), int(arr.shape[1] / 41)
        for i in range(len(HRss)):
            w_ = int(i/w)
            h_ = i-w_*w
        #    new_img[w_*41:(1+w_)*41, h_*41:(1+h_)*41] += HRss[i]
        for j in range(int(arr.shape[0] / 41)):
            for k in range(int(arr.shape[1] / 41)):
                #print(j, k, j*int(arr.shape[1] / 41)+k)
                new_img[41*j:41*(j+1), 41*k:41*(k+1)] += HRss[j*int(arr.shape[1] / 41)+k]
        #print(new_img)
        #a1 = arr[:w*41, :h*41]*255.
        a1 = arr[:w*41, :h*41]
        a2 = new_img[:w*41, :h*41].astype(float)
        print(PSNR(a1, ori_arr[:41*w, :41*h]))
        print(PSNR(a2, ori_arr[:41*w, :41*h]))
        psnrSum += PSNR(a2, ori_arr[:41*w, :41*h])
        #print(a1.shape)
        #print(a2.shape)
        w, h = a1.shape
        cnt = 0.0
    print("平均信噪比")
    print(psnrSum/9.)
    #for i in range(w):
    #    for j in range(h):
    #       cnt += abs(a1[i, j]-a2[i, j])
    #print(cnt)
    #print("arr")
    #print(arr[:w*41, :h*41])
    #print("new_img")
    #print(new_img[:w*41, :h*41])
    #plt.subplot(2, 2, 1)
    #plt.imshow(Image.fromarray(a1), cmap='gray')
    #plt.subplot(2, 2, 2)
    #plt.imshow(Image.fromarray(a2), cmap='gray')
    #plt.subplot(2, 2, 3)
    #plt.imshow(Image.fromarray(ori_arr[:41*w, :41*h]), cmap='gray')
    #plt.show()
