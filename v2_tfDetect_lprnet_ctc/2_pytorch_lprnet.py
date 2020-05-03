"""
@author Vadim Placinta

comments are in romanian
"""

# I think $ is gonna be null label, it should be last
alphabet = [' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
            'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '$']
# max plate len also should be common
max_plate_len = 15

# partea pt incarcarea datelor
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import cv2
import random
import numpy as np
import time

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def labels_to_text(labels):
    return ''.join(list(map(lambda x: alphabet[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: alphabet.index(x), text))

def apply_random_effect(img, eff):
    shape = (img.shape[1], img.shape[0])
    if eff == 0:  # apply noise
        noise = cv2.imread('noise.jpg')
        noise = cv2.resize(noise, shape)
        alpha = random.randint(75, 100) / 100  # between 0.55 and 0.9 is good
        return cv2.addWeighted(img, alpha, noise, (1.0 - alpha), 0.0)
    elif eff == 1:  # apply dirt
        dirt = cv2.imread('dirt.jpg')
        dirt = cv2.resize(dirt, shape)
        alpha = random.randint(75, 100) / 100  # between 0.55 and 0.9 is good
        return cv2.addWeighted(img, alpha, dirt, (1.0 - alpha), 0.0)
    else:  # apply blur
        return cv2.blur(img, (random.randint(1, 2), random.randint(1, 2)))  # every param in range 1:4

class BatchGenerator():
    def __init__(self, dir, img_w, img_h, alphabet, max_plate_len, batch_size, shuffle=False):
        self.dir = dir
        self.img_w = img_w
        self.img_h = img_h
        self.alphabet = alphabet
        self.max_plate_len = max_plate_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        data = pd.read_csv(dir + "dataset.csv")
        links = data.iloc[:, 0].values
        plt_text = data.iloc[:, 1].values

        if len(links) != len(plt_text):
            print("EROARE: Numarul de imagini nu coincide cu cel de texte!!!")
            return

        self.nr_sampels = len(links)
        self.imgs = np.zeros((self.nr_sampels, img_h, img_w, 3), dtype=np.uint8)
        self.texts = []

        for i, l in enumerate(links):
            img = cv2.imread(dir + l)
            self.imgs[i] = img
        for i, t in enumerate(plt_text):
            self.texts.append(t)

        self.indexes = torch.randperm(self.nr_sampels)
        self.cnt = 0

    def next_sample(self):
        self.cnt += 1
        if self.cnt >= self.nr_sampels:
            self.cnt = 0
            if self.shuffle:
                self.indexes = torch.randperm(self.nr_sampels)
        return self.imgs[self.indexes[self.cnt]], self.texts[self.indexes[self.cnt]]

    def next_batch(self):
        global T
        X_data = torch.zeros((self.batch_size, 3, self.img_h, self.img_w)).to(dev)
        Y_data = torch.zeros((self.batch_size, self.max_plate_len), dtype=torch.long).to(dev)
        # nr de timestamp care ies din nn, e nevoie pentru ctc_loss
        X_data_len = torch.ones(self.batch_size, dtype=torch.long) * T
        Y_data_len = torch.zeros(self.batch_size, dtype=torch.long)

        X_data_len.to(dev)
        Y_data_len.to(dev)

        for i in range(self.batch_size):
            img, text = self.next_sample()
            img = apply_random_effect(img, random.randint(1,3))
            img = torch.from_numpy(img)
            img.type(torch.float32)
            img //= 255
            X_data[i] = img.permute(2, 0, 1)
            Y_data[i] = torch.from_numpy(np.array(text_to_labels(text) + [1]*(self.max_plate_len-len(text))))
            Y_data_len[i] = len(text)

        return X_data, Y_data, X_data_len, Y_data_len

# import json
# import os
#
# class BatchGenerator2():
#     def __init__(self, img_w, img_h, alphabet, max_plate_len, batch_size, shuffle=False):
#         self.img_w = img_w
#         self.img_h = img_h
#         self.alphabet = alphabet
#         self.max_plate_len = max_plate_len
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#         dir = "/home/vadim/Downloads/train_lpr/"
#         im_names = os.listdir(dir+"img/")
#
#         self.nr_sampels = len(im_names)
#         self.imgs = np.zeros((self.nr_sampels, img_h, img_w, 3), dtype=np.uint8)
#         self.texts = []
#
#         for i, im in enumerate(im_names):
#             img = cv2.imread(dir+"/img/"+im)
#             img = cv2.resize(img, (img_w, img_h))
#             self.imgs[i] = img
#
#             f = open(dir+"/ann/"+im+".json")
#             data = json.load(f)
#             self.texts.append(data["description"])
#
#         self.indexes = torch.randperm(self.nr_sampels)
#         self.cnt = 0
#
#     def next_sample(self):
#         self.cnt += 1
#         if self.cnt >= self.nr_sampels:
#             self.cnt = 0
#             if self.shuffle:
#                 self.indexes = torch.randperm(self.nr_sampels)
#         return self.imgs[self.indexes[self.cnt]], self.texts[self.indexes[self.cnt]]
#
#     def next_batch(self):
#         global T
#         X_data = torch.zeros((self.batch_size, 3, self.img_h, self.img_w))
#         Y_data = torch.zeros((self.batch_size, self.max_plate_len), dtype=torch.long)
#         # nr de timestamp care ies din nn, e nevoie pentru ctc_loss
#         X_data_len = torch.ones(self.batch_size, dtype=torch.long) * T
#         Y_data_len = torch.zeros(self.batch_size, dtype=torch.long)
#
#         for i in range(self.batch_size):
#             img, text = self.next_sample()
#             #img = apply_random_effect(img, random.randint(1,3))
#             img = torch.from_numpy(img)
#             img.type(torch.float32)
#             img /= 255
#             X_data[i] = img.permute(2, 0, 1)
#             Y_data[i] = torch.from_numpy(np.array(text_to_labels(text) + [1]*(self.max_plate_len-len(text))))
#             Y_data_len[i] = len(text)
#
#         return X_data, Y_data, X_data_len, Y_data_len

#partea unde construiesc reteaua

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #se aplica pt torch.Size([1, 30, 1, 74])
        return torch.nn.functional.log_softmax(x, 3)

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # se aplica pt torch.Size([1, 30, 1, 74])
        #pentru CTCloss
        #trebuie sa fie (T,N,C) , where T=input length, N=batch size, C=number of classes
        x = x.permute(3, 0, 1, 2)
        return x.view(x.shape[0], x.shape[1], x.shape[2])


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(ch_out//4, ch_out, kernel_size=1)
        )
        self.block.apply(init_weights)

    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.lprnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 1)),
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(4,2,1)),
            nn.Dropout(0.5),
            nn.Conv2d(64, 256, kernel_size=(4,1), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, class_num, kernel_size=(1,13), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(), # torch.Size([1, 30, 1, 74])
            Softmax(),
            Reshape()
        )
        self.lprnet.apply(init_weights)

    def forward(self, x):
        return self.lprnet(x)



train_dir = 'plate_and_nr_dataset/train/'
valid_dir = 'plate_and_nr_dataset/validation/'
train_set = BatchGenerator(train_dir, 94, 24, alphabet, max_plate_len, 32, shuffle=True)
valid_set = BatchGenerator(valid_dir, 94, 24, alphabet, max_plate_len, 32)
# train_set = BatchGenerator2(94, 24, alphabet, max_plate_len, 32, shuffle=True)
# valid_set = BatchGenerator2(94, 24, alphabet, max_plate_len, 32)

T = 74 #input sequence length

ctc_loss = nn.CTCLoss(blank=alphabet.index('$'))

# x_test = torch.rand((10, 3, 24, 94))
# l = LPRNet(len(alphabet))
# y = l(x_test)
# print(y.shape)

model = LPRNet(len(alphabet))
model.to(dev)
opt = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(opt, step_size=100000, gamma=0.1)

def train(epochs):
    for i in range(epochs):
        model.train()
        start = time.time()
        X_data, Y_data, X_data_len, Y_data_len = train_set.next_batch()

        X_data.to(dev)
        Y_data.to(dev)
        X_data_len.to(dev)
        Y_data_len.to(dev)

        X_data = model(X_data)
        loss = ctc_loss(X_data, Y_data, X_data_len, Y_data_len)
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()
        print("-- epoch --", i)
        print(loss)
        print(1/(time.time()-start),"FPS")

        if i % 10000 == 0:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, "checkpoint.tar")

def load_checkpoint():
    checkpoint = torch.load("checkpoint.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("epoch nr ={}, last loss = {}".format(epoch,loss))
    model.eval()

train(250000)