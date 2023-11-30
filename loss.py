import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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

    def transform(self, x):
        B, C, W, H = x.shape
        x = x.reshape(B, C // group_size, -1)

        return x

    def kld(self, pred, soft):
        _, _, h, w = pred.size()
        _, _, H, W = soft.size()
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
