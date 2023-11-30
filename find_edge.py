import PIL.Image as Image
from PIL import ImageFilter
from torchvision.transforms import transforms as t
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2


class early_edge(nn.Module):
    def __init__(self, mode='std'):
        super().__init__()
        self.mode = mode
        self.edge = nn.Sequential(nn.Conv2d(3, 1, 7, 2, 3),
                                  nn.BatchNorm2d(1),
                                  nn.ReLU()
                                  )
        self.k = torch.tensor([[[[-1. / 8, -1. / 8, -1. / 8], [-1. / 8, 1., -1. / 8], [-1. / 8, -1. / 8, -1. / 8]]]],
                              requires_grad=False, device='cuda:0')
        self.sobel = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]],
                                  requires_grad=False, device='cuda:0')

    def std_edge(self, input):
        scale = 1
        B, C, H, W = input.shape
        windows = input.unfold(2, 2, 1).unfold(3, 2, 1)
        windows = windows.contiguous().view(B, C, H - 1, W - 1, 4)
        edge = torch.std(windows, dim=-1) * scale
        edge = nn.functional.interpolate(edge, size=(H, W))

        return edge

    def forward(self, input):

        if self.mode == 'conv':
            input = input.mean(dim=1).unsqueeze(1)
            out = self.edge(input)
        elif self.mode == 'std':
            input = input.mean(dim=1).unsqueeze(1)
            out = self.std_edge(input)
        elif self.mode == 'laplacian':
            input = input.mean(dim=1).unsqueeze(1)
            out = nn.functional.conv2d(input, self.k, stride=1, padding=1)
        elif self.mode == 'sobel':
            input = input.mean(dim=1).unsqueeze(1)
            out = nn.functional.conv2d(input, self.sobel, stride=1, padding=1) + \
                  nn.functional.conv2d(input, self.sobel.transpose(-1, -2), stride=1, padding=1)
        return out

# t2p = t.ToPILImage()
# p2t = t.ToTensor()
# find_edge = early_edge('std')
# path = '/home/xyx/code/NYUDepthV2/test/depth/412.png'
# img_p = Image.open(path)
#
# img_t = p2t(img_p).unsqueeze(0)
# edge = find_edge(img_t)
# edge = t2p(edge.squeeze(0))
# edge.show()
