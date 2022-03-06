import pandas as pd
import numpy as np
import math
import time

st = time.time()
dft = pd.read_csv('../train.csv')
df = dft.values.tolist()
wh = [] # 所有真实框的宽高信息
for i in range(len(df)):
    bbox = df[i][3].replace('[', '').replace(']', '').split(', ')
    wh.append([float(bbox[2]), float(bbox[3])])

def cas_iou(box, centers):
    width = np.minimum(centers[:, 0], box[0])
    height = np.minimum(centers[:, 1], box[1])

    Intersection = width * height
    Union = centers[:, 0] * centers[:, 1] + box[0] * box[1] - Intersection

    iou = Intersection / Union
    d = 1 - iou
    num = np.argmin(d)

    return num

def KMeans(boxes, k):
    row = boxes.shape[0]
    nums = np.full(row, -1)

    np.random.rand()

    centers = boxes[np.random.choice(row, k, replace=False)]
    last_clu = nums

    while True:

        for i in range(row):
            nums[i] = cas_iou(boxes[i], centers)

        if (last_clu == nums).all():
            break

        for j in range(k):
            centers[j] = np.median(boxes[nums == j], axis=0)

        last_clu = nums

    return centers

data = np.array(wh)
out = KMeans(data, 9)
anchors = out[np.argsort(out[:, 0])]

with open('yolo_wheat_anchors.txt', 'w') as f:
    for i, anchor in enumerate(anchors):
        if i == 0:
            x_y = '%d, %d' % (anchor[0], anchor[1])
        else:
            x_y = ',  %d, %d' % (anchor[0], anchor[1])
        f.write(x_y)

# 3个特征层，3个先验框

end = time.time()
print('共耗时：%.2f s' % (end-st))