import math

import torch
import torch.nn as nn

from toolbox.loss import MscKDLoss


class CA(nn.Module):
    def __init__(self, channel=64, reduction=8):
        super(CA, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channel, channel, 7, 2, 3, groups=channel),
                                  nn.BatchNorm2d(channel),
                                  nn.Conv2d(channel, channel, 7, 2, 3, groups=channel),
                                  nn.BatchNorm2d(channel),
                                  nn.Conv2d(channel, channel, 7, 2, 3, groups=channel),
                                  nn.BatchNorm2d(channel),
                                  # nn.Conv2d(channel, channel // 16, 1),
                                  # nn.Conv2d(channel // 16, channel // 16, 1),
                                  # nn.Conv2d(channel // 16, channel, 1),
                                  )
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.bn = nn.BatchNorm2d(channel)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.conv(x)
        return y


class SA(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(SA, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(channel, channel // 8, 1),
                                   nn.BatchNorm2d(channel // 8),
                                   )

    def forward(self, x):
        max_out = self.conv0(x)
        return max_out


class RB(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.convc = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.convs = nn.Conv2d(channel // 8, channel, 1)
        self.convf = nn.Conv2d(channel*2, channel, 3, 1, 1)

    def forward(self, fc, fs):
        h, w = fs.shape[2:]
        fc = self.convc(nn.functional.interpolate(fc, size=(h, w), mode='bilinear'))
        fs = self.convs(fs)
        f = self.convf(torch.cat((fc, fs), dim=1))
        return f


class attention_kd(nn.Module):
    def __init__(self, c=[64, 64, 64, 64]):
        # 64, 64, 64, 64  64, 128, 320, 512
        super().__init__()
        # self.channel_s = nn.ModuleList()
        self.channel_t = nn.ModuleList()
        # self.spatial_s = nn.ModuleList()
        self.spatial_t = nn.ModuleList()
        # self.rebuild_s = nn.ModuleList()
        self.rebuild_t = nn.ModuleList()
        self.norm_s = nn.ModuleList()
        self.norm_t = nn.ModuleList()
        for i in c:
            # self.channel_s.append(CA(i))
            self.channel_t.append(CA(i))
            # self.spatial_s.append(SA(i))
            self.spatial_t.append(SA(i))
            # self.rebuild_s.append(RB(i))
            self.rebuild_t.append(RB(i))

        self.kld = MscKDLoss()

    def rebuild_loss(self, f_s):
        rb_loss_s = 0
        l = len(f_s)
        for i in range(l):
            s = f_s[i]
            c_s = self.channel_t[i](s)
            s_s = self.spatial_t[i](s)
            rb_s = self.rebuild_t[i](c_s, s_s)
            loss_s = self.kld(rb_s, s)
            rb_loss_s += loss_s
        return rb_loss_s / l

    def forward(self, f_s):
        rb_loss_s = self.rebuild_loss(f_s)

        return rb_loss_s


if __name__ == '__main__':
    a = torch.randn(4, 64, 30, 40)
    b = torch.randn(4, 64, 30, 40)
    c = torch.randn(4, 64, 60, 80)
    d = torch.randn(4, 64, 60, 80)
    s = [a, b]
    t = [c, d]
    l = attention_kd([64, 64])
    loss = l(s)
    print(loss)
