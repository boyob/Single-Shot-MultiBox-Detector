import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.utilsConfig import Config
from utils.utilsNetL2Norm import L2Norm
from utils.utilsNetDefaultBox import DefaultBox
from utils.utilsNetDetect import Detect


class Net(torch.nn.Module):
    def __init__(self, phase):
        super(Net, self).__init__()
        self.phase = phase
        self.buildParts()

    def buildParts(self):
        mbox = [4, 6, 6, 6, 4, 4]
        vgg_layers, extra_layers = self.add_vgg(3), self.add_extras(1024)
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg_layers[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(vgg_layers[v].out_channels, mbox[k] * Config["num_classes"], kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * Config["num_classes"], kernel_size=3, padding=1)]
        self.vgg = nn.ModuleList(vgg_layers)
        self.L2Norm = L2Norm(512, 20)
        self.extra = nn.ModuleList(extra_layers)
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.defaultBox = DefaultBox(Config)
        with torch.no_grad():
            self.defaults = Variable(self.defaultBox.forward())
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # self.detect = Detect(Config["num_classes"], 0, 200, 0.01, 0.45)
            # self.softmax = nn.Softmax()
            self.detect = Detect()

    def forward(self, x):
        # 1.构建数据流图
        sources = list()
        loc = list()
        conf = list()
        # 1.1四个卷积块
        for k in range(23):
            x = self.vgg[k](x)
        # 1.2一个正则化
        s = self.L2Norm(x)
        sources.append(s)
        # 1.3两个卷积块
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        # 1.4四个卷积块
        for k, v in enumerate(self.extra):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # 1.5回归层和分类层
        for (x, l, c) in zip(sources, self.loc, self.conf):  # 它们3个的shape都是(6,)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # 2.resize
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, Config['num_classes'])
        # 3.
        if self.phase == "test":
            # output = self.detect(loc, self.softmax(conf), self.defaults)
            output = self.detect.apply(Config["num_classes"], 0, 200, 0.01, 0.45,
                                       loc, self.softmax(conf), self.defaults)
        else:
            output = (loc, conf, self.defaults)
        return output

    def add_vgg(self, i):
        # VGG-16的5个卷积块
        base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'T']
        layers = []
        in_channels = i
        for v in base:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            elif v == 'T':
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        fc7_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        fc7_2 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [fc7_1, nn.ReLU(inplace=True), fc7_2, nn.ReLU(inplace=True)]
        return layers

    def add_extras(self, i):
        layers = []
        in_channels = i
        # Block 6：19,19,1024 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
        # Block 7：10,10,512 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        # Block 8：5,5,256 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        # Block 9：3,3,256 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        return layers
