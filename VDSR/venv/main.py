from model import Net
import dataloader
from torch.utils.data import DataLoader
from PIL import Image
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorboard import summary

import os


Maxepoch = 201 # 最大epoch数
net = Net().cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: "+str(device))
print(net)
#xs, ys = dataloader.get_xy()
dataset = dataloader.loadh5()
loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
test_img = Image.open("imgs_256to512/01.jpg")
test_img = test_img.convert("L")
ori_img = Image.open("imgs_512/01.jpg")
ori_img = ori_img.convert("L")
#arr = np.array(test_img) / 255.0  # 归一
#ori_arr = np.array(ori_img) / 255.0
arr = np.array(test_img)  # 不归一
ori_arr = np.array(ori_img)
testxs = dataloader.split_img([arr])
orixs =dataloader.split_img([ori_arr])
criterion = nn.MSELoss()

train_log_dir = 'logs/train/'
test_log_dir = 'logs/test/'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)



#if len(xs) != len(ys):
#    print("error: xs != ys")
#l = len(xs)

# 训练
def train(epoch, net):
    lr = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #lr = 0.0001
    #optimizer = torch.optim.Adam(net.parameters())
    losssum = 0
    for e in range(epoch+1, Maxepoch):
        lr = lr*(0.2**(e//10))
        #for param_group in optimizer.param_groups:
            #param_group["lr"] = lr
        for iter, batch in enumerate(loader, 1):
            x, y = batch[0], batch[1]
            x = torch.autograd.Variable(x).cuda()
            y = torch.autograd.Variable(y, requires_grad=False).cuda()

            output = net(x)
            loss = criterion(output, y).to(device)
            losssum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.4)
            optimizer.step()
            if iter % 10 == 0:
                print("iter: "+str(iter)+" "+str(losssum)+" ")
                losssum = 0
        testLoss = accuracy()
        print("epoch: "+str(e)+" "+str(loss.item())+" "+str(testLoss))
        # 统计数据
        with train_summary_writer.as_default():
            summary.scalar('loss', loss.item(), e)
        with test_summary_writer.as_default():
            summary.scalar('loss', testLoss, e)

        if e % 10 == 0:
            path = "checkpoint/"+"model_epoch_{}.pth".format(e)
            state = {"epcoh":e, "model":net}
            torch.save(state, path)


# 测试单张图片上的Loss
def accuracy():
    Lsum = 0
    with torch.no_grad():
        for i in range(len(testxs)):
            each = testxs[i]
            each = torch.tensor(each).float().cuda()
            each = torch.autograd.Variable(each).view(-1, 1, each.shape[0], each.shape[1])
            output = net(each)
            y = orixs[i]
            y = torch.tensor(y).float().cuda()
            y = torch.autograd.Variable(y).view(-1, 1, y.shape[0], y.shape[1])
            loss = criterion(output, y).to(device)
            Lsum += loss.item()

    return Lsum

if __name__ == '__main__':
    checkpoint = torch.load("checkpoint/model_epoch_50.pth")
    #print(checkpoint)
    #net.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    net = checkpoint['model']
    epoch = checkpoint['epcoh']

    #epoch = 0
    train(epoch, net)


    '''
        for iter, batch in enumerate(loader, 1):
        x, y = batch[0], batch[1]
        #print(x.numpy().shape)
        ximg = Image.fromarray(x.numpy()[0, 0]*255.)
        yimg = Image.fromarray(y.numpy()[0, 0]*255.)
        plt.subplot(2, 1, 1)
        plt.imshow(ximg, cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(yimg, cmap='gray')
        plt.show()
    img = Image.open("imgs_256/01.jpg")
    img = img.convert("L")
    print(img.size)
    bmg = img.resize((img.size[0]*2, img.size[1]*2))
    bmg.save("checkpoint/fortest.jpg")
    plt.imshow(bmg, cmap='gray')
    plt.show()
    '''
