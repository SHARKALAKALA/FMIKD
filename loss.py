import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from toolbox.lavaszSoftmax import lovasz_softmax


# med_frq = [0.000000, 0.452448, 0.637584, 0.377464, 0.585595,
#            0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
#            2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
#            0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
#            1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
#            4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
#            3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
#            0.750738, 4.040773,2.154782,0.771605,0.781544,0.377464]

class lovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=None):
        super(lovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.classes = classes

    def forward(self, output, target):
        if not isinstance(output, tuple):
            output = (output,)
        loss = 0
        for item in output:
            h, w = item.size(2), item.size(3)
            # 变换大小需要4维
            label = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            logits = F.softmax(item, dim=1)
            loss += lovasz_softmax(logits, label.squeeze(1), ignore=self.ignore_index, per_image=self.per_image,
                                   classes=self.classes)
        return loss / len(output)


class MscBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MscBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, (tuple, list)):
            input = (input,)
        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.float(), size=(h, w), mode='nearest')
            loss += F.binary_cross_entropy_with_logits(item, item_target, reduction=self.reduction)
        return loss / len(input)


class MscCrossEntropyLoss(nn.Module):
    #

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', gate_gt=None):
        super(MscCrossEntropyLoss, self).__init__()

        self.weight = weight
        self.gate_gt = gate_gt
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        # weight = [0.2,0.4,0.6,0.8]

        # att,w = target.size()[1:]

        for item in input:
            h, w = item.size(2), item.size(3)

            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))

            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                                    ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


def FLoss(prediction, target):
    log_like = True
    beta = 1
    target = torch.nn.functional.interpolate(target.float(), size=prediction.shape[-2:], mode='nearest')
    EPS = 1e-10
    N = prediction.size(0)
    TP = (prediction * target).view(N, -1).sum(dim=1)
    H = beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
    fmeasure = (1 + beta) * TP / (H + EPS)
    if log_like:
        floss = -torch.log(fmeasure)
    else:
        floss = (1 - fmeasure)
    return floss.mean()


class EdgeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EdgeLoss, self).__init__()

    def FLoss(prediction, target):
        log_like = True
        beta = 1
        target = torch.nn.functional.interpolate(target.float(), size=prediction.shape[-2:], mode='nearest')
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + beta) * TP / (H + EPS)
        if log_like:
            floss = -torch.log(fmeasure)
        else:
            floss = (1 - fmeasure)
        return floss.mean()

    def forward(self, input, target):
        if not isinstance(input, (tuple, list)):
            input = (input,)
        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.float(), size=(h, w), mode='nearest')
            f_loss = FLoss(torch.sigmoid(item), item_target)
            f_loss = torch.zeros(1, device='cuda:0') if f_loss == torch.tensor(torch.inf, device='cuda:0') else f_loss
            loss += f_loss
        return loss / len(input)


class MscKDLoss(nn.Module):
    def __init__(self, temperature=1):
        super(MscKDLoss, self).__init__()
        self.KLD = torch.nn.KLDivLoss(reduction='sum')
        self.temperature = temperature

    def transform(self, x, loss_type='channel'):
        B, C, W, H = x.shape
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = 1
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def kld(self, pred, soft):
        _, _, h, w = pred.size()
        _, _, H, W = soft.size()
        # pred = F.interpolate(pred, size=(H, W), mode='bilinear')
        # soft = F.interpolate(soft, size=(att, w), mode='bilinear')
        p_s = self.transform(pred)
        p_t = self.transform(soft)
        p_s = F.log_softmax(p_s / self.temperature, dim=-1)
        p_t = F.softmax(p_t / self.temperature, dim=-1)
        loss = self.KLD(p_s, p_t) / (p_s.numel() / p_s.shape[-1])
        return loss

    def forward(self, input, target):
        if not isinstance(input, (tuple, list)):
            input = (input,)
            target = (target, )
        loss = 0
        for item, item_target in zip(input, target):
            h, w = item.size(2), item.size(3)
            # if item.shape != item_target.shape:
            #     item_target = F.interpolate(target, size=(att, w), mode='bilinear')
            loss += self.kld(item, item_target)
        return loss / len(input)


if __name__ == '__main__':
    x = torch.randn(2, 2)
    print(x)
    out = x.mean(1)
    # import torch
    # ll = 'layer3_1 '
    # out = ll.split('_1')[0]+ll.split('_1')[1]
    print(out)
    # depth = torch.randn(6,3,480,640)
    # score = torch.Tensor(6,1)
    # print(score.shape)
    # print(score)
    # score = score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640)
    # # out = torch.mul(depth,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640))
    # print(score.shape)
    # print(score)
    # torch.randn(6,3,480,640)
    # print(out)
    # out = out.view(3,480,640)
    # print(out)

    # predict = torch.randn((2, 21, 512, 512))
    # gt = torch.randint(0, 255, (2, 512, 512))

    # loss_function = MscCrossEntropyLoss()
    # result = loss_function(predict, gt)
    # print(result)
