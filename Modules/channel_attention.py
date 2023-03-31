##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 通道注意力代码，对应框架CA模块
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch.nn as nn
import torch
import torch.nn.functional as F
#from .layer import MultiSpectralAttentionLayer
# # SE block add to U-net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)


class RCA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(RCA, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1x1 = nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=1,stride=1,padding=0,groups=1,bias=False)
        self.avgPool = nn.AvgPool2d(2,stride=2)
        self.maxPool = nn.MaxPool2d(2,stride=2)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv4 = conv3x3(planes, planes * 2)
        self.bn4 = nn.BatchNorm2d(planes * 2)
        self.conv5 = conv3x3(planes * 2, planes)
        self.bn5 = nn.BatchNorm2d(planes)


        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes <= 16:
            self.globalAvgPool = nn.AvgPool2d((224, 300), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 150), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((112, 150), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AdaptiveAvgPool2d(1) # (56, 75) for ISIC2018
            self.globalMaxPool = nn.AdaptiveAvgPool2d(1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((28, 37), stride=1)  # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((28, 37), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((14, 18), stride=1)  # (14, 18) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_r = self.conv1x1(residual)
        out_r = self.bn1(out_r)
        out_r = self.relu(out_r)

        residual_b = out + out_r

        residual_a = self.avgPool(residual_b)
        residual_m = self.maxPool(residual_b)
        residual_a = self.conv2(residual_a)
        residual_a = self.bn2(residual_a)
        residual_a = self.conv3(residual_a)
        residual_a = self.bn3(residual_a)
        residual_m = self.conv4(residual_m)
        residual_m = self.bn4(residual_m)
        residual_m = self.conv5(residual_m)
        residual_m = self.bn5(residual_m)

        residual_all = residual_a+residual_m
        residual_all = self.globalAvgPool(residual_all)
        residual_all = self.sigmoid(residual_all)
        out = out*residual_all

        return out,out
