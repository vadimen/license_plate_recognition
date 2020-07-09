"""
@author Vadim Placinta
am folosit ca exemplu:
https://github.com/sirius-ai/LPRNet_Pytorch
https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/image_ocr.ipynb
"""
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import cv2
import random
import numpy as np
import time
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

# I think '-' is gonna be null label, it should be last
alphabet = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
            '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
            '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
            '新',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', 'I', 'O', '-'
            ]
# max plate len also should be common
max_plate_len = 15

#detectăm dispozitivul pe care se va rula codul
device = torch.device(
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

#alfabete folosite pentru a decoda informatia din imaginile din CCPD dataset
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
# partea pt incarcarea datelor, versiunea pentru setul mic taiat manual din imagini
# class BatchGenerator():
#     def __init__(self, dir, img_w, img_h, alphabet, max_plate_len, batch_size, shuffle=False):
#         self.dir = dir
#         self.img_w = img_w
#         self.img_h = img_h
#         self.alphabet = alphabet
#         self.max_plate_len = max_plate_len
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#         data = pd.read_csv(dir + "dataset.csv")
#         links = data.iloc[:, 0].values
#         plt_text = data.iloc[:, 1].values
#
#         if len(links) != len(plt_text):
#             print("EROARE: Numarul de imagini nu coincide cu cel de texte!!!")
#             return
#
#         self.nr_sampels = len(links)
#         self.imgs = np.zeros((self.nr_sampels, img_h, img_w, 3), dtype=np.uint8)
#         self.texts = []
#
#         for i, l in enumerate(links):
#             img = cv2.imread(dir + l)
#             self.imgs[i] = img
#         for i, t in enumerate(plt_text):
#             self.texts.append(t)
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
#         X_data = torch.zeros((self.batch_size, 3, self.img_h, self.img_w)).to(device)
#         Y_data = torch.zeros((self.batch_size, self.max_plate_len), dtype=torch.long).to(device)
#         # nr de timestamp care ies din nn, e nevoie pentru ctc_loss
#         X_data_len = torch.ones(self.batch_size, dtype=torch.long) * T
#         Y_data_len = torch.zeros(self.batch_size, dtype=torch.long)
#
#         X_data_len.to(device)
#         Y_data_len.to(device)
#
#         for i in range(self.batch_size):
#             img, text = self.next_sample()
#             img = apply_random_effect(img, random.randint(1,3))
#             img = torch.from_numpy(img)
#             img.type(torch.float32)
#             img //= 255
#             X_data[i] = img.permute(2, 0, 1)
#             Y_data[i] = torch.from_numpy(np.array(text_to_labels(text) + [1]*(self.max_plate_len-len(text))))
#             Y_data_len[i] = len(text)
#
#         return X_data, Y_data, X_data_len, Y_data_len

# versiunea pentru CCPD dataset
class BatchGenerator():
    def __init__(self, dir, img_w, img_h, alphabet,
 		max_plate_len, batch_size, shuffle=False):
        self.dir = dir
        self.img_w = img_w
        self.img_h = img_h
        self.alphabet = alphabet
        self.max_plate_len = max_plate_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        image_names = os.listdir(dir)

        self.nr_sampels = len(image_names)
        self.imgs = torch.zeros((self.nr_sampels, img_h, img_w, 3), dtype=torch.uint8)
        self.texts = []

        for i,n in enumerate(image_names):
            img = cv2.imread(dir + n)
            self.imgs[i] = torch.from_numpy(img)
            #plt.figure()
            #plt.imshow(img)
            #plt.figure()
            #plt.imshow(self.imgs[i])
            # construirea textului
            plate_text = n.split('.')[0].split('-')[1].split('_')
            plate_text[0] = provinces[int(plate_text[0])]
            plate_text[1] = alphabets[int(plate_text[1])]
            for j in range(2, len(plate_text)):
                plate_text[j] = ads[int(plate_text[j])]
            self.texts.append("".join(plate_text))

        self.indexes = range(self.nr_sampels) #torch.randperm(self.nr_sampels)
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
        X_data = torch.zeros((self.batch_size, 3, self.img_h, self.img_w)).to(device)
        Y_data = torch.zeros((self.batch_size, self.max_plate_len), dtype=torch.long).to(device)
        # nr de timestamp care ies din nn, e nevoie pentru ctc_loss
        X_data_len = torch.ones(self.batch_size, dtype=torch.long) * T
        Y_data_len = torch.zeros(self.batch_size, dtype=torch.long)

        for i in range(self.batch_size):
            img, text = self.next_sample()
            #img = apply_random_effect(img, random.randint(1, 3))
            #img = torch.from_numpy(img)
            img = img.type(torch.float32)
	        #normalizăm pixelii
            img /= 255
	        #pytorch cere ca nr de canale să fie primul
            X_data[i] = img.permute(2, 0, 1)
            Y_data[i] = torch.from_numpy(np.array(text_to_labels(text) + [1] * (self.max_plate_len - len(text))))
            Y_data_len[i] = len(text)
        return X_data, Y_data, X_data_len, Y_data_len

# partea unde construim reteaua
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
        # se aplica pt torch.Size([1, 30, 1, 74]) --> 30 nu mai e 30, am schimbat alfabet
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

train_dir = '/home/vadim/Documents/CCPD2019/ccpd_train/'
valid_dir = '/home/vadim/Documents/CCPD2019/ccpd_validation/'
train_set = BatchGenerator(train_dir, 94, 24, alphabet, max_plate_len, 32, shuffle=True)
valid_set = BatchGenerator(valid_dir, 94, 24, alphabet, max_plate_len, 32)

T = 74  # input sequence length

lprnet = LPRNet(class_num=len(alphabet))

def train_one_epoch(model, log_interval, epoch, loader, optimizer, device):
    model.to(device)
    model.train()
    N_count = 0
    losses = []
    start = time.time()
    for i in range(loader.nr_sampels//loader.batch_size+1):
        X_data, Y_data, X_data_len, Y_data_len = loader.next_batch()
        N_count += X_data.size(0)

        X_data = model(X_data)
        optimizer.zero_grad()
        loss = ctc_loss(X_data, Y_data, X_data_len, Y_data_len)
        if loss.item() == np.inf:
            continue
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss {:.6f}'.format(
                epoch+1, N_count, loader.nr_sampels, (N_count/loader.nr_sampels) * 100.0,
                loss.item()
            ))

    print("{:.2f}s time taken to train for epoch {}".format(time.time() - start, epoch+1))
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': np.mean(losses)
    }, "lprnet_chckpnt.tar")
    print("lprnet_chckpnt_epoch_{}.tar successfully saved".format(epoch+1))

    return np.mean(losses)

def validation_one_epoch(model, epoch, loader, device):
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for i in range(loader.nr_sampels // loader.batch_size + 1):
            X_data, Y_data, X_data_len, Y_data_len = loader.next_batch()

            X_data = model(X_data)
            loss = ctc_loss(X_data, Y_data, X_data_len, Y_data_len)
            if loss.item() == np.inf:
                continue
            losses.append(loss.item())

    print("Validation epoch:", epoch+1, ", loss:", np.mean(losses))
    return np.mean(losses)

optimizer = optim.Adam(lprnet.parameters())
ctc_loss = nn.CTCLoss(blank=alphabet.index('-'), reduction='mean')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)

train_losses = []
validation_losses = []
def train(epochs):
    log_interval = 2
    for epoch in range(epochs):
        tl = train_one_epoch(lprnet, log_interval, epoch, train_set, optimizer, device)
        vl = validation_one_epoch(lprnet, epoch, valid_set, device)
        train_losses.append(tl)
        validation_losses.append(vl)
    np.save("train_losses.npy", train_losses)
    np.save('validation_losses.npy', validation_losses)
    print("Train and Validation losses were saved.")

def load_checkpoint():
    checkpoint = torch.load("/home/vadim/Downloads/lprnet_chckpnt.tar", map_location=torch.device('cpu'))
    lprnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("epoch nr ={}, last loss = {}".format(epoch, loss))
    lprnet.eval()

# TESTING
import itertools

#calculates min edit distance in nr of replace, add, edit operation
def min_distance(word1, word2):
    rows = len(word1) + 1
    cols = len(word2) + 1
    arr = np.zeros((rows, cols), dtype=np.uint8)
    arr[0] = range(cols)
    arr[:, 0] = range(rows)

    for i in range(1, rows):
        for j in range(1, cols):
            if word1[i - 1] == word2[j - 1]:
                arr[i, j] = arr[i - 1, j - 1]
            else:
                m = min(arr[i - 1, j - 1], arr[i, j - 1], arr[i - 1, j])
                arr[i, j] = m + 1

    return arr[rows - 1][cols - 1]

def decode_batch(out):
    ret = []
    bs = out.shape[0]
    for i in range(bs):
        out_best = list(np.argmax(out[i, :], 1))
        out_best = [k for k, _ in itertools.groupby(out_best)]
        st = ''.join(list(map(
                lambda x: alphabet[int(x)] if int(x) in range(alphabet.index('新'),alphabet.index('-')) else '',
                out_best)))
        ret.append(st)
    return ret

def test():
    X_data, Y_data, X_data_len, Y_data_len = valid_set.next_batch()
    bs = X_data.shape[0]
    net_out_value = lprnet(X_data)
    net_out_value = net_out_value.permute(0, 2, 1)

    pred_texts = decode_batch(net_out_value.detach().numpy())

    texts = []
    for i,label in enumerate(Y_data):
        text = labels_to_text(label)
        texts.append(text[0:Y_data_len[i]])

    label_errors = []
    for i in range(bs):
        print('Predicted: %s    True: %s' % (pred_texts[i], texts[i]))
        label_errors.append(min_distance(pred_texts[i], texts[i]))
        #img = X_data[i].permute(1,2,0).detach().numpy()
        #cv2.imshow('img', img)

        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    break

    cv2.destroyAllWindows()
    return np.mean(label_errors)

def test_validation_dataset():
    errs = []
    for i in range(10):
        errs.append(test())
    print(np.mean(errs))