import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import math

class yolodataset(Dataset):
    def __init__(self, images, targets, input_shape, num_classes, anchors, train=True, anchors_mask=((6,7,8), (3,4,5), (0,1,2)), times=(32, 16, 8), cuda=True):
        super(yolodataset, self).__init__()
        self.images = images
        self.targets = targets
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors.view(9, 2) # [9, 2] tensor
        self.anchors_mask = anchors_mask
        self.times = times
        self.cuda = cuda
        self.eps = 1e-6
        self.train = train

    def __getitem__(self, index):
        tx = self.images[index]
        ty = self.targets[index]
        if self.train:
            x, ty = self.image_preprocess(tx, ty)
        else:
            x = self.image_preprocess_test(tx)
        # print(f'x_shape:{x.shape}')
        y = self.getlabels(ty) # y : python list --> [tensor0, tensor1, tensor2]
        # print(f'y_shape:{y[0].shape}')

        return x, y

    def __len__(self):
        return len(self.images)

    def getlabels(self, labels):
        '''
        :param labels: python_list --> [num_gt, 5]
        :param label: python_list --> [x, y, w, h, cls]
        :return:
        '''
        # print(torch.tensor(labels).shape)
        if self.input_shape%32 != 0:
            raise RuntimeError('input shape can not be Divisible by 32')
        l_sizes = [self.input_shape//time for time in self.times]
        y_true = [torch.zeros(3, l_size, l_size, 5+self.num_classes) for l_size in l_sizes]
        if len(labels) == 0:
            return y_true
        # 对于每个label 计算其与其他9个先验框的iou值
        labelswh = torch.tensor(labels)[..., 2:4]
        labelswh = torch.unsqueeze(labelswh, dim=-2).expand(labelswh.shape[0], self.anchors.shape[0], 2)
        wh = torch.minimum(labelswh, self.anchors)
        inter = wh[..., 0] * wh[..., 1]
        union = labelswh[..., 0] * labelswh[..., 1] + self.anchors[..., 0] * self.anchors[..., 1] - inter
        iou = inter/torch.clamp(union, self.eps) # [n, 9]

        best_anchor = torch.argmax(iou, dim=-1)
        # print(best_anchor)
        # idx 第几个先验框，iou为当前先验框与当前box的iou值
        for label_i, anchor_i in enumerate(best_anchor):
            # f = False
            # for anchor_i in anchors_i:
            #     f = False
            for l, anchor_mask in enumerate(self.anchors_mask):
                if anchor_i not in anchor_mask:
                    continue

                l_size = y_true[l].shape[1]

                coff = self.input_shape/1024
                # print(l_size)
                stride = self.input_shape/l_size
                m, n = labels[label_i][0]*coff//stride, labels[label_i][1]*coff//stride
                # dx, dy = (labels[label_i][0] - sw*m)/sw, (labels[label_i][1] - sw*n)/sw
                # tw, ty = torch.log(labels[label_i][2]/self.anchors[anchor_i][0]),  torch.log(labels[label_i][3]/self.anchors[anchor_i][1])

                idxt, m, n = int(anchor_i%3), int(m), int(n)
                # if y_true[l][idxt, n, m, 0] == 1:
                #     break
                y_true[l][idxt, n, m, 0] = labels[label_i][0] * coff # cx
                y_true[l][idxt, n, m, 1] = labels[label_i][1] * coff # cy
                y_true[l][idxt, n, m, 2] = labels[label_i][2] * coff # w
                y_true[l][idxt, n, m, 3] = labels[label_i][3] * coff # h
                y_true[l][idxt, n, m, 4] = 1 # conf
                y_true[l][idxt, n, m, 5+int(labels[label_i][4])] = 1 # cls
                # f = True
                # break
                # if f:
                #     break
            #     f = True
            #     break
            # if f == True:
            #     break
        return y_true

    # def cal_iou(self, box1, box2):
    #     '''
    #     :param box1: [w, h]
    #     :param box2: [w, h]
    #     :return: float
    #     '''
    #     w = min(box1[0], box2[0])
    #     h = min(box1[1], box2[1])
    #     Intersection = w*h
    #     Union = box1[0]*box1[1] + box2[0]*box2[1] - Intersection
    #     return Intersection/max(Union, 1e-6)

    def image_preprocess(self, img_path, labels):
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.5)(img)
        img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)
        img = np.transpose(np.array(img)/255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        # img, labels = self.horizontal_flip(img, labels)
        return img_tensor, labels

    def image_preprocess_test(self, img_path):
        img = Image.open(img_path).convert('RGB')
        # img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.5)(img)
        img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)
        img = np.transpose(np.array(img) / 255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        # img, labels = self.horizontal_flip(img, labels)
        return img_tensor
        # img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)
        # img = np.array(img, dtype=np.float32)/255.
        # img = np.transpose(img, (2, 0, 1))
        #
        # return torch.from_numpy(img).type(torch.FloatTensor)
        # return img_path
        # return torch.tensor(img)
