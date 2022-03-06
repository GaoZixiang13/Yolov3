
import pandas as pd
import torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob, tqdm
from PIL import Image

def fit_one_epoch(model, optimizer, loss_func, lr_scheduler, EPOCH, epoch, train_loader, val_loader, RE_SIZE_shape, val_loss_save, time, CUDA, warmup=False):
    loss, loss2 = 0, 0
    with tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{EPOCH} Training', postfix=dict) as pbar:
        model.train()
        for iter1, (bx, by) in enumerate(train_loader):
            if CUDA:
                bx = bx.type(torch.FloatTensor).cuda() # [bs, 3, input, input]
                by = [y.type(torch.FloatTensor).cuda() for y in by] # [bs, num_gt, 5]

            # bs = bx.size(0)
            optimizer.zero_grad()
            train_loss = 0
            preds = model(bx)
            train_loss += loss_func(preds, by)

            # train_loss = train_loss_tol/torch.maximum(num_gt, torch.ones_like(num_gt))
            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            # print(cnt1)
            # print(f'loss:{loss}, index:{i+1}, even_loss:{loss/(i+1)}')

            pbar.set_postfix(**{'loss': loss / (iter1 + 1),
                                # 'loss ciou': lc1,
                                # 'loss conf': lc2,
                                # 'loss cls': lc3,
                                'lr': optimizer.param_groups[0]['lr']
                                })
            pbar.update(1)

    lr_scheduler.step()
    training_loss = loss / (iter1 + 1)
    print('Finish Training')
    print('Start Validation')
    # model.eval()
    # device_val = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    with tqdm.tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{EPOCH} Validation', postfix=dict) as pbar:
        model.eval()
        with torch.no_grad():
            for iter2, (bx, by) in enumerate(val_loader):
                if CUDA:
                    bx = bx.type(torch.FloatTensor).cuda()
                    by = [y.type(torch.FloatTensor).cuda() for y in by]

                val_loss = 0
                preds = model(bx)
                val_loss += loss_func(preds, by).item()

                # val_loss = val_loss_tol/max(num_gt, 1)
                loss2 += val_loss

                pbar.set_postfix(**{'val_loss': loss2 / (iter2 + 1),
                                    })
                pbar.update(1)

    if loss2 / (iter2 + 1) < val_loss_save:
        if warmup==False:
            torch.save(model.state_dict(), '/home/b201/gzx/yolov3_self/logs/val_loss%.3f-size%03d-lr%.8f-ep%03d-loss%.3f.pth' % (loss2 / (iter2 + 1), RE_SIZE_shape, optimizer.param_groups[0]['lr'], epoch + 1, training_loss))
            val_loss_save = loss2 / (iter2 + 1)
            print('Model state_dict save success!')
    print(f'Training loss:{training_loss}, || Validation loss:{loss2 / (iter2 + 1)}')

    return val_loss_save




