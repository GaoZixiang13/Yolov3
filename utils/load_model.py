
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_model(model, model_path):
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)

    # print(model_dict.keys())
    # print(pretrained_dict.keys())
    # print(pretrained_dict)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    # model_dict.update(pretrained_dict)
    # print(pretrained_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def load_anchors(anchor_path):
    with open(anchor_path) as f:
        line = f.readline().split(',')
        # print(line)
        ret, tmp1, tmp2 = [], [], []
        for i in range(1, len(line)+1):
            tmp1.append(int(line[i-1]))

            if i % 2 == 0:
                tmp2.append(tmp1)
                tmp1 = []
            if i % 6 == 0:
                ret.append(tmp2)
                tmp2 = []

        return ret

