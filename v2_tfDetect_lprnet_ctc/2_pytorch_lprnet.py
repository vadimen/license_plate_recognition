"""
@author Vadim Placinta

comments are in romanian
"""

# I think $ is gonna be null label, it should be last
alphabet = [' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
            'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '$']
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
from torch.autograd import Variable

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
            img = apply_random_effect(img, random.randint(1, 3))
            # cv2.imshow("1", img)
            img = torch.from_numpy(img)
            # cv2.imshow("2", img.numpy())
            img = img.type(torch.float32)
            # cv2.imshow("3", img.numpy())
            img /= 255
            # cv2.imshow("4", img.numpy())
            X_data[i] = img.permute(2, 0, 1)
            Y_data[i] = torch.from_numpy(np.array(text_to_labels(text) + [1] * (self.max_plate_len - len(text))))
            Y_data_len[i] = len(text)

        return X_data, Y_data, X_data_len, Y_data_len


# partea unde construiesc reteaua
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate=0.5, phase=True):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


train_dir = 'plate_and_nr_dataset/train/'
valid_dir = 'plate_and_nr_dataset/validation/'
train_set = BatchGenerator(train_dir, 94, 24, alphabet, max_plate_len, 32, shuffle=True)
valid_set = BatchGenerator(valid_dir, 94, 24, alphabet, max_plate_len, 32)

T = 18  # input sequence length

lprnet = LPRNet(class_num=len(alphabet) + 1, lpr_max_len=max_plate_len)
lprnet.to(dev)


def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def xavier(param):
    nn.init.xavier_uniform(param)


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = xavier(1)
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0.01


lprnet.backbone.apply(weights_init)
lprnet.container.apply(weights_init)
print("initial net weights successful!")

opt = optim.RMSprop(lprnet.parameters(), lr=0.1, alpha=0.9, eps=1e-08,
                    momentum=0.9, weight_decay=2e-5)
ctc_loss = nn.CTCLoss(blank=alphabet.index('$'), reduction='mean')


def train(epochs):
    for i in range(epochs):
        start = time.time()

        X_data, Y_data, X_data_len, Y_data_len = train_set.next_batch()
        lr = adjust_learning_rate(opt, i, 0.1, [4, 8, 12, 14, 16])

        logits = lprnet(X_data)
        log_probs = logits.permute(2, 0, 1)  # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()
        opt.zero_grad()
        loss = ctc_loss(log_probs, Y_data, X_data_len, Y_data_len)
        if loss.item() == np.inf:
            continue
        loss.backward()
        opt.step()
        print("epoch:", i, "loss:", loss.item())
        print((1 / (time.time() - start)), "FPS")

        if i % 1000 == 0:
            torch.save({
                'epoch': i,
                'model_state_dict': lprnet.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss.item()
            }, "checkpoint.tar")


def load_checkpoint():
    checkpoint = torch.load("checkpoint-sirius-ai.tar", map_location=torch.device('cpu'))
    lprnet.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("epoch nr ={}, last loss = {}".format(epoch, loss))
    lprnet.eval()


# TESTING
import itertools

def decode_batch(out):
    ret = []
    bs = out.shape[0]
    for i in range(bs):
        out_best = list(np.argmax(out[i, :], 1))
        out_best = [k for k, _ in itertools.groupby(out_best)]
        st = ''.join(list(map(lambda x: alphabet[int(x)] if int(x) < len(alphabet) - 1 else '', out_best)))
        ret.append(st)
    return ret


def test():
    X_data, Y_data, X_data_len, Y_data_len = valid_set.next_batch()
    bs = X_data.shape[0]
    net_out_value = lprnet(X_data)
    net_out_value = net_out_value.permute(0, 2, 1)

    pred_texts = decode_batch(net_out_value.detach().numpy())

    texts = []
    for label in Y_data:
        text = labels_to_text(label)
        texts.append(text)

    for i in range(bs):
        print('Predicted: %s    True: %s' % (pred_texts[i], texts[i]))
        img = X_data[i].permute(1,2,0).detach().numpy()
        cv2.imshow('img', img)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


#sirius-ai test
# from PIL import Image, ImageDraw, ImageFont
#
# def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
#     if (isinstance(img, np.ndarray)):  # detect opencv format or not
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img)
#     fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
#     draw.text(pos, text, textColor, font=fontText)
#
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#
# def show(img, label, target):
#     img = np.transpose(img, (1, 2, 0))
#     img *= 128.
#     img += 127.5
#     img = img.astype(np.uint8)
#
#     lb = ""
#     for i in label:
#         lb += alphabet[i]
#     tg = ""
#     for j in target.tolist():
#         tg += alphabet[int(j)]
#
#     flag = "F"
#     if lb == tg:
#         flag = "T"
#     # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
#     img = cv2ImgAddText(img, lb, (0, 0))
#     cv2.imshow("test", img)
#     print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# def test():
#     Tp = 0
#     Tn_1 = 0
#     Tn_2 = 0
#     t1 = time.time()
#     X_data, Y_data, _, _ = train_set.next_batch()
#     bs = X_data.shape[0]
#
#     # forward
#     prebs = lprnet(X_data)
#     # greedy decode
#     prebs = prebs.detach().numpy()
#     preb_labels = list()
#     for i in range(prebs.shape[0]):
#         preb = prebs[i, :, :]
#         preb_label = list()
#         for j in range(preb.shape[1]):
#             preb_label.append(np.argmax(preb[:, j], axis=0))
#         no_repeat_blank_label = list()
#         pre_c = preb_label[0]
#         if pre_c != len(alphabet) - 1:
#             no_repeat_blank_label.append(pre_c)
#         for c in preb_label: # dropout repeate label and blank label
#             if (pre_c == c) or (c == len(alphabet) - 1):
#                 if c == len(alphabet) - 1:
#                     pre_c = c
#                 continue
#             no_repeat_blank_label.append(c)
#             pre_c = c
#         preb_labels.append(no_repeat_blank_label)
#     for i, label in enumerate(preb_labels):
#         # show image and its predict label
#         if True:
#             show(X_data[i].detach().numpy(), label, Y_data[i])
#         if len(label) != len(Y_data[i]):
#             Tn_1 += 1
#             continue
#         if (np.asarray(targets[i]) == np.asarray(label)).all():
#             Tp += 1
#         else:
#             Tn_2 += 1
#     Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
#     print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
#     t2 = time.time()
#     print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / 50, 50))
#
#     cv2.destroyAllWindows()
