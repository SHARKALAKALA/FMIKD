import math

import torch
import torch.nn as nn
from thop import profile

from find_edge import early_edge
from toolbox.models import module as m

from toolbox.models.decoders import Unet, light
from toolbox.models.encoder.mix_transformer import mit_b5, mit_b4, mit_b1, mit_b2, mit_b3


class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = mit_b2()

    def forward(self, rgb):
        fea_list = self.rgb(rgb)
        return fea_list


class dee(nn.Module):
    def __init__(self, r_c):
        super().__init__()
        self.r1 = m.SA_conv(r_c)
        self.d_down = nn.Conv2d(1, 1, 3, 2, 1)
        self.edge = nn.Conv2d(2, 1, 3, 1, 1)
        self.q = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                               nn.BatchNorm2d(1),
                               nn.ReLU())
        self.k = nn.Sequential(nn.Conv2d(r_c, 1, 3, 1, 1),
                               nn.BatchNorm2d(1),
                               nn.ReLU())
        self.v = nn.Sequential(nn.Conv2d(r_c, r_c, 3, 1, 1),
                               nn.BatchNorm2d(r_c),
                               nn.ReLU())

    def forward(self, d, r):
        r1 = self.r1(r)
        d = self.d_down(d)
        edge = self.edge(torch.cat((d, r1), dim=1))
        att = self.q(d) * self.k(r)
        att = torch.sigmoid(att + torch.sigmoid(edge) * att)
        r_out = att * self.v(r) + r
        return edge, r_out


class edge_fea(nn.Module):
    def __init__(self, f_c):
        super().__init__()
        self.embeding = nn.Sequential(nn.Conv2d(1, 1, 7, 2, 3),
                                      nn.BatchNorm2d(1),
                                      nn.ReLU())
        self.dee1 = dee(f_c[0])
        self.dee2 = dee(f_c[1])
        self.dee3 = dee(f_c[2])
        self.dee4 = dee(f_c[3])

    def forward(self, rgb, fea_list):
        d = self.embeding(rgb)
        d1, f1 = self.dee1(d, fea_list[0])
        d2, f2 = self.dee2(d1, fea_list[1])
        d3, f3 = self.dee3(d2, fea_list[2])
        d4, f4 = self.dee4(d3, fea_list[3])
        return [d1, d2, d3, d4], [f1, f2, f3, f4]


class edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, e_l):
        if e_l[0].shape[2:] != e_l[1].shape[2:]:
            e_l[1] = self.up2(e_l[1])
            e_l[2] = self.up4(e_l[2])
            e_l[3] = self.up8(e_l[3])
        edge = self.dw(torch.cat(e_l, dim=1))
        e_l.append(edge)
        return e_l


class head(nn.Module):
    def __init__(self, f_c):
        super().__init__()
        self.c0 = nn.Conv2d(f_c[0], f_c[0], 1)
        self.c1 = nn.Conv2d(f_c[1], f_c[0], 1)
        self.c2 = nn.Conv2d(f_c[2], f_c[0], 1)
        self.c3 = nn.Conv2d(f_c[3], f_c[0], 1)
        self.fin = nn.Conv2d(f_c[0], f_c[0], 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, fea_list):
        if fea_list[0].shape[2:] != fea_list[1].shape[2:]:
            f0 = fea_list[0]
            f1 = self.up2(fea_list[1])
            f2 = self.up4(fea_list[2])
            f3 = self.up8(fea_list[3])
        f0 = self.c0(f0)
        f1 = self.c1(f1)
        f2 = self.c2(f2)
        f3 = self.c3(f3)
        fin = self.fin(f0 + f1 + f2 + f3)
        return fin, [f0, f1, f2, f3]


class seg_decoder(nn.Module):
    def __init__(self, inc):
        super(seg_decoder, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(inc[3], inc[2], 1, 1)
        self.conv2 = nn.Conv2d(inc[2], inc[1], 1, 1)
        self.conv3 = nn.Conv2d(inc[1], inc[0], 1, 1)
        self.conv4 = nn.Conv2d(inc[0], inc[0], 1, 1)

    def forward(self, fea_list):
        p1 = self.conv1(fea_list[3])
        p2 = self.conv2(fea_list[2] + self.up2(p1))
        p3 = self.conv3(fea_list[1] + self.up2(p2))
        p4 = self.conv4(fea_list[0] + self.up2(p3))

        return p4


class seg_seg(nn.Module):
    def __init__(self, f_c=[64, 128, 320, 512], num_classes=41):
        super(seg_seg, self).__init__()
        # 32, 64, 160, 256   64, 128, 320, 512  64, 128, 256, 512  48, 96, 192, 384  40, 80, 160, 320
        self.en = encoder()
        self.d = early_edge('std')
        self.de = seg_decoder(f_c)
        self.edge_fea = edge_fea(f_c)
        self.edge = edge()
        self.head = head(f_c)
        self.fin = nn.Conv2d(f_c[0], num_classes, 3, 1, 1)
        self.ed = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                nn.BatchNorm2d(1),
                                nn.ReLU())
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, rgb, dep=None):
        fea_list = self.en(rgb)
        d = self.d(rgb)
        e_l, fea_list_r = self.edge_fea(d, fea_list)

        edge = self.edge(e_l)
        seg, fea = self.head(fea_list_r)
        seg = torch.sigmoid(edge[-1]) * seg + seg
        pred = self.up4(self.fin(seg))
        return pred, edge


if __name__ == '__main__':
    a = torch.randn(1, 3, 480, 640)
    b = torch.randn(1, 1, 480, 640)
    model = seg_seg()
    model.load_state_dict(torch.load('/home/xyx/model3/run/nyu_student_b2/model_best.pth'))
    # model.cuda()
    # model.eval()
    # s = model(a, b)
    # for i in s:
    #     print(i.shape)
    flops, params = profile(model, (a, b))
    print('Flops: ', flops / 1e9, 'G')
    print('Params: ', params / 1e6, 'M')