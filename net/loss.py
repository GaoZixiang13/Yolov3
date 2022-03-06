import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class yolov3_loss(nn.Module):
    def __init__(self, input_shape, num_classes, anchors, noobj_ignore=0.5, label_smoothing=0, times=(32, 16, 8), cuda=True):
        super(yolov3_loss, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.times = times
        self.cuda = cuda
        self.ignore = noobj_ignore
        self.label_smoothing = label_smoothing

    #  将t中的所有值变动到[t_min, t_max]
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCEloss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    # def BCEloss(self, pred, target):
    #     loss = nn.BCELoss()
    #     return loss(pred, target)

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    # def CIOU(self, box1, box2):
    #     '''
    #     :param box1: [..., 4] (t,x,y,w,h,...1...)
    #     :param box2:
    #     :return:
    #     '''
    #     # box1 *= self.input_shape
    #     # box2 *= self.input_shape
    #     eps = 1e-6
    #     # print(box1.shape)
    #     # print(box2.shape)
    #     # print(box1.size())
    #     # print(box2.size())
    #     zsxy = torch.maximum(box1[..., 0:2] - box1[..., 2:4]/2, box2[..., 0:2] - box2[..., 2:4]/2)
    #     yxxy = torch.minimum(box1[..., 0:2] + box1[..., 2:4]/2, box2[..., 0:2] + box2[..., 2:4]/2)
    #     wh = torch.maximum(yxxy - zsxy, torch.zeros_like(yxxy))
    #     Intersection = wh[..., 0]*wh[..., 1]
    #     Union = box1[..., 2]*box1[..., 3] + box2[..., 2]*box2[..., 3] - Intersection
    #     iou = Intersection/torch.clamp(Union, min=1e-6)
    #     # print(torch.sum(iou>1))
    #     v = (4 / (math.pi ** 2)) * torch.pow(
    #         torch.atan(box1[..., 2] / torch.clamp(box1[..., 3], min=1e-6)) - torch.atan(box2[..., 2] / torch.clamp(box2[..., 3], min=1e-6)), 2)
    #     alpha = v/torch.clamp((1-iou+v), min=1e-6)
    #
    #     xy1 = torch.minimum(box1[..., 0:2] - box1[..., 2:4]/2, box2[..., 0:2] - box2[..., 2:4]/2)
    #     xy2 = torch.maximum(box1[..., 0:2] + box1[..., 2:4]/2, box2[..., 0:2] + box2[..., 2:4]/2)
    #     dis = torch.maximum(xy2 - xy1, torch.zeros_like(xy1))
    #     c = torch.sum(torch.pow(dis, 2), dim=-1)
    #
    #     ciou = iou - alpha * v - 1.0*torch.sum(torch.pow(box1[..., 0:2] - box2[..., 0:2], 2), dim=-1) / torch.clamp(c, min=1e-6)
    #
    #     return ciou

    def decode(self, x, y, w, h, bs, l_size, l):
        '''
        :param x: tensor: [3, l_size, l_size]
        :param y:
        :param w:
        :param h:
        :return: [..., 4]
        '''
        ix = torch.linspace(0, l_size-1, l_size).repeat(l_size, 1).repeat(bs * 3, 1, 1).view(x.shape).cuda()
        iy = torch.linspace(0, l_size-1, l_size).repeat(l_size, 1).t().repeat(bs * 3, 1, 1).view(y.shape).cuda()
        x = (ix+x)/l_size
        y = (iy+y)/l_size
        anchorw = self.anchors[[2, 1, 0]][..., 0][l].repeat(bs, 1).unsqueeze(-1).repeat(1, 1, l_size*l_size)\
            .view(3, l_size, l_size).cuda()
        anchorh = self.anchors[[2, 1, 0]][..., 1][l].repeat(bs, 1).unsqueeze(-1).repeat(1, 1, l_size*l_size)\
            .view(3, l_size, l_size).cuda()
        w = anchorw * torch.exp(w)
        # w = w.squeeze(0)
        h = anchorh * torch.exp(h)
        # h = h.squeeze(0)
        # print(x.size())
        # print(y.size())
        # print(w.size())
        # print(h.size())
        box = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], dim=-1)
        return box

    def forward(self, predictions, targets):
        '''
        :param preds: torch tensor --> [batch_size, 3*(5+num_classes), l_size, l_size]
        :param target: torch tensor -->
        :return:
        '''
        # print(torch.sum(target[..., 0] == 1))
        l_sizes = [self.input_shape//time for time in self.times]
        bs = predictions[0].size(0)

        loss_reg, loss_conf, loss_cls = 0, 0, 0
        for b in range(bs):
            # preds, gt_true = [], []
            for l, prediction_s in enumerate(predictions):
                gt_trues = targets[l][b][targets[l][b][..., 4] == 1].view(-1, 5+self.num_classes)
                l_size = l_sizes[l]
                # print(f'l size:{l_size}')
                # print(f'nums:{torch.sum(target[..., 0] == 1)}')
                # print(prediction.shape)
                # print(target.shape)
                # loss, num_gt = torch.tensor(0), torch.tensor(0)
                # print(prediction.shape)
                prediction = prediction_s[b].view(3, 5+self.num_classes, l_size, l_size).permute(0, 2, 3, 1).contiguous() #[3,l,l,6]
                predw, predh = prediction[..., 2], prediction[..., 3]
                prediction = torch.sigmoid(prediction)
                conf, cls = prediction[..., 4], prediction[..., 5:]
                predx, predy = prediction[..., 0], prediction[..., 1]
                box_pred = self.decode(predx, predy, predw, predh, 1, l_size, l)
                if gt_trues.size(0) == 0:
                    # loss_conf += self.BCEloss(conf, torch.zeros_like(conf).cuda())
                    loss_conf += torch.sum(self.BCEloss(conf, 0))
                    continue
                # print(box_pred.size())
                # print(conf.size())
                # print(cls.size())
                preds_boxes = torch.cat([box_pred, conf.unsqueeze(-1), cls], dim=-1).view(-1, 5+self.num_classes)
                # preds.append(pred_decoded)
                # preds_boxes = torch.cat(preds, dim=0) # [m, 6]
                # gt_trues = torch.cat(gt_true, dim=0) # [n, 6]
                # allocated_mask = torch.zeros(preds_boxes.size(0)).cuda()
                gt_trues_t = torch.unsqueeze(gt_trues, dim=-2).expand(gt_trues.size(0), preds_boxes.size(0), 5+self.num_classes)
                # 这里计算正样本的损失
                iou = self.cal_iou(gt_trues_t, preds_boxes)
                best_pred_indices = torch.argmax(iou, dim=-1)
                    # if allocated_mask[indice] == 0:
                # print(preds_boxes[best_pred_indices][..., 0].size())
                # print(gt_trues[..., 0].size())
                # preds_gt_boxes = preds_boxes[best_pred_indices]
                loss_x = self.MSELoss(preds_boxes[best_pred_indices][..., 0], gt_trues[..., 0])
                loss_y = self.MSELoss(preds_boxes[best_pred_indices][..., 1], gt_trues[..., 1])
                loss_w = self.MSELoss(preds_boxes[best_pred_indices][..., 2], gt_trues[..., 2])
                loss_h = self.MSELoss(preds_boxes[best_pred_indices][..., 3], gt_trues[..., 3])
                loss_reg_s = (loss_x + loss_y + loss_w + loss_h) * (2 - gt_trues[..., 2] * gt_trues[..., 3])

                # tar_cls = torch.zeros(self.num_classes).cuda()
                # tar_cls[int(gt_trues[..., 4])] = 1
                tar_cls = self.smooth_labels(gt_trues[..., 5:], self.label_smoothing, self.num_classes)
                loss_cls_s = self.BCEloss(preds_boxes[best_pred_indices][..., 5:], tar_cls)

                noobj_mask = torch.ones(preds_boxes.size(0))
                noobj_mask[best_pred_indices] = 0
                preds_box_tmp = torch.unsqueeze(preds_boxes, dim=-2).expand(preds_boxes.size(0), gt_trues.size(0),
                                                                            5 + self.num_classes)
                iou1 = self.cal_iou(preds_box_tmp, gt_trues)
                maxiou1 = torch.max(iou1, dim=-1).values
                noobj_mask[maxiou1 >= self.ignore] = 0

                # print(preds_boxes.size())
                # print(preds_boxes[best_pred_indices][..., 4].size())
                # preds_true_mask = torch.zeros(preds_boxes.size())
                # print(preds_boxes[..., 4][best_pred_indices].size())
                # print(best_pred_indices.size())
                # conf_true = torch.ones_like(best_pred_indices).cuda()
                # conf_false = torch.zeros(torch.sum(noobj_mask==1).item()).cuda()
                # print(preds_boxes[best_pred_indices][..., 4].size())
                # print(conf_true.size())
                # loss_conf_s = self.BCEloss(preds_boxes[noobj_mask==1][..., 4], conf_false)
                # loss_conf_s = self.BCEloss(preds_boxes[noobj_mask==1][..., 4], 0)
                # print(preds_boxes)
                # print(best_pred_indices)
                # print(preds_boxes[best_pred_indices])
                # print(preds_boxes[best_pred_indices][..., 4])
                loss_conf_s = torch.sum(self.BCEloss(preds_boxes[best_pred_indices][..., 4], 1)) + \
                              torch.sum(self.BCEloss(preds_boxes[noobj_mask==1][..., 4], 0))

                loss_reg += torch.sum(loss_reg_s)
                loss_conf += torch.sum(loss_conf_s)
                loss_cls += torch.sum(loss_cls_s)
                # allocated_mask[indice] = 1
                # break

        loss = loss_reg + loss_conf + loss_cls

        return loss

    def cal_iou(self, boxes1, box2):
        '''
        :param boxes1:
        :param box2:
        :return:
        '''
        zsxy = torch.maximum(boxes1[..., 0:2] - boxes1[..., 2:4]/2, box2[..., 0:2] - box2[..., 2:4]/2)
        yxxy = torch.minimum(boxes1[..., 0:2] + boxes1[..., 2:4]/2, box2[..., 0:2] + box2[..., 2:4]/2)
        wh = torch.maximum(yxxy - zsxy, torch.zeros_like(yxxy))
        Intersection = wh[..., 0] * wh[..., 1]
        Union = boxes1[..., 2] * boxes1[..., 3] + box2[..., 2] * box2[..., 3] - Intersection

        return Intersection/torch.clamp(Union, 1e-6)


