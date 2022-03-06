import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob, tqdm
from PIL import Image

df = pd.read_csv('/home/b201/gzx/global-wheat-detection/train.csv')
df = df.values.tolist()
# id = df['image_id'].unique()
# dict1, dict2 = {}, {}
# for i, image_id in enumerate(id):
#     dict1[image_id] = i
#     dict2[i] = image_id
# print(len(id))
name_set = set()
for path in glob.glob('/home/b201/gzx/yolov4-pytorch-master/VOCdevkit/VOC2007/JPEGImages/*.jpg'):
    list = path.split('/')
    name = list[len(list)-1].split('.')[0]
    name_set.add(name)

x = []
y = []
dict = {}
for i in range(len(df)):
    bbox = df[i][3].replace('[', '').replace(']', '').split(', ')
    # print(bbox)
    name = str(df[i][0])
    if name not in name_set:
        continue
    x1, y1, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx, cy = x1 + w/2, y1 + h/2
    sx = '/home/b201/gzx/yolov4-pytorch-master/VOCdevkit/VOC2007/JPEGImages/' + name + '.jpg'
    # cx, cy, w, h = cx/input_shape, cy/input_shape, w/input_shape, h/input_shape
    c = 0  # 该框所属的类别，需根据标签文件指定
    sy = [cx, cy, w, h, c]
    if sx not in dict.keys():
        dict[sx] = []
    dict[sx].append(sy)

cnt = 0
for name, labels in dict.items():
    x.append(name)
    if len(labels) == 0:
        cnt += 1
    y.append(labels)

print(len(x))
print(len(y))
print(f'无标签图片数量：{cnt}')

# x_trainVal, x_test, y_trainVal, y_test = train_test_split(x, y, test_size=0.05, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True)


with open('/home/b201/gzx/yolov3_self/train.txt', 'w') as f:
    for i, data in enumerate(x_train):
        f.write(data + ' ')
        for y in y_train[i]:
            for z in y:
                f.write(str(z) + ' ')
        f.write('\n')

with open('/home/b201/gzx/yolov3_self/val.txt', 'w') as f:
    for i, data in enumerate(x_val):
        f.write(data + ' ')
        for y in y_val[i]:
            for z in y:
                f.write(str(z) + ' ')
        f.write('\n')

# with open('/home/b201/gzx/yolov4_self_0/test.txt', 'w') as f:
#     for i, data in enumerate(x_test):
#         f.write(data + ' ')
#         for y in y_test[i]:
#             for z in y:
#                 f.write(str(z) + ' ')
#         f.write('\n')

print('text文件创建完成！')
