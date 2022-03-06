from net import loss
from net import yolov3
from net import preprocess
from utils import load_model, utils_fit

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob, tqdm
from PIL import Image, ImageDraw
from torchvision.ops import nms

x_test_path = glob.glob('/home/b201/gzx/yolov4_self_0/test/*.jpg')
print(x_test_path)

CUDA = True
NMS_hold = 0.45
conf_hold = 0.5
pic_shape = 1024
anchors_mask = ((6,7,8), (3,4,5), (0,1,2))
input_shape = 640

anchors_path = '/home/b201/gzx/yolov3_self/utils/yolo_wheat_anchors.txt'
# 先验框的大小
# 输入为416，anchor大小为
anchors = load_model.load_anchors(anchors_path)
anchors = torch.tensor(anchors)*input_shape/pic_shape

model = yolov3.yolov3(1)
model_path = '/home/b201/gzx/yolov3_self/logs/val_loss1.843-size640-lr0.00024099-ep026-loss1.514.pth'
load_model.load_model(model, model_path)

if CUDA:
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    anchors = anchors.cuda()

def Nms_yolov4_self(bboxes, threshold=0.3):
    # area = bboxes[..., 3] * bboxes[..., 4]
    # classes = torch.argmax(bboxes[..., 5:])
    zs = bboxes[..., 0:2] - bboxes[..., 2:4]/2
    yx = bboxes[..., 0:2] + bboxes[..., 2:4]/2
    box4 = torch.cat([zs, yx], dim=-1)
    # bboxes[..., 0:2], bboxes[..., 2:4] = zs, yx
    # boxes = bboxes.view(-1, 6)
    # boxes = torch.cat((zs, yx, conf.unsqueeze(-1), cls.unsqueeze(-1)), dim=-1).view(-1, 6) # [n, 6]

    indice = nms(box4, bboxes[..., 4]*bboxes[..., 5], threshold)
    # boxes = torch.cat((bboxes[..., 5].unsqueeze(dim=-1), zs, bboxes[..., 2:4]), dim=-1)

    return box4[indice] # [m, 4]

def image_preprocess_test(img_path):
    img = Image.open(img_path).convert('RGB')
    # img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.5)(img)
    img = img.resize((input_shape, input_shape), Image.BICUBIC)
    img = np.transpose(np.array(img) / 255., (2, 0, 1))
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # img, labels = self.horizontal_flip(img, labels)
    return img_tensor

def decode(x, y, w, h, bs, l_size, l, stride):
    '''
    :param x: tensor:
    :param y:
    :param w:
    :param h:
    :return:
    '''
    ix = torch.linspace(0, l_size-1, l_size).repeat(l_size, 1).repeat(bs * 3, 1, 1).view(x.shape).cuda()
    iy = torch.linspace(0, l_size-1, l_size).repeat(l_size, 1).t().repeat(bs * 3, 1, 1).view(y.shape).cuda()
    x = (ix+x) * stride
    y = (iy+y) * stride
    anchorw = anchors[[2, 1, 0]][..., 0][l].repeat(bs, 1).unsqueeze(-1).repeat(1, 1, l_size * l_size) \
        .view(bs, 3, l_size, l_size).cuda()
    anchorh = anchors[[2, 1, 0]][..., 1][l].repeat(bs, 1).unsqueeze(-1).repeat(1, 1, l_size * l_size) \
        .view(bs, 3, l_size, l_size).cuda()
    # print(anchorw.shape)
    # print(w.shape)
    w = anchorw * torch.exp(w)
    h = anchorh * torch.exp(h)
    box = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], dim=-1)

    return box.cuda()

model.eval()
for i, path in enumerate(x_test_path):
    print(path)
    jpgname = path.split('/')[-1]
    img = image_preprocess_test(path).type(torch.FloatTensor).unsqueeze(0)
    # print(img.shape)
    if CUDA:
        img = img.cuda()

    outputs = model(img)
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    predboxes = []
    for j, output in enumerate(outputs):
        l_size = output.size(-1)
        output = output.view(1, 3, 6, l_size, l_size).permute(0, 1, 3, 4, 2).contiguous()

        conf = torch.sigmoid(output[..., 4])
        cls = torch.sigmoid(output[..., 5])
        outw = output[..., 2]
        outh = output[..., 3]
        outx = torch.sigmoid(output[..., 0])
        outy = torch.sigmoid(output[..., 1])

        # print(outx)
        # print(outy)
        # print(outw)
        # print(outh)
        pred_boxt = decode(outx, outy, outw, outh, 1, l_size, j, input_shape/l_size)
        # print(pred_boxt)
        pred_box = torch.cat([pred_boxt, conf.unsqueeze(-1), cls.unsqueeze(-1)], dim=-1)
        '''
        过滤掉class-specific confidence score低于阈值的框
        '''
        predboxes.append(pred_box.view(-1, 6))

    predboxes = torch.cat(predboxes, dim=0)
    '''
    对过滤之后得到的框进行非极大抑制得到最后的预测框
    '''
    predboxes = predboxes[predboxes[..., 4] * predboxes[..., 5] >= conf_hold]
    boxes = Nms_yolov4_self(predboxes.view(-1, 6), threshold=NMS_hold) # [m, 5] [cls, zsx, zsy, w, h]
    boxes[..., :4] *= pic_shape/input_shape
    '''
    最后得到的这个预测框可以用来进行绘制或是计算P、R等信息
    '''
    for b in boxes:
        b = b.tolist()
        draw.rectangle([b[0], b[1], b[2], b[3]], outline='red', width=2)

    image.save('/home/b201/gzx/yolov3_self/predict_img/' + jpgname)

