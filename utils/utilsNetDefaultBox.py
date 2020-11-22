import torch
import numpy as np
from math import sqrt as sqrt


# 生成8732个默认框的坐标，返回值output.shape=[8732, 4]。
# output[i]=[cx, cy, w, h], 坐标值要乘300才是原图上的像素坐标。
class DefaultBox(object):
    def __init__(self, cfg):
        super(DefaultBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

    def forward(self):
        mean = []
        # 在原图上划分栅格。
        for k, f in enumerate(self.feature_maps):
            x, y = np.meshgrid(np.arange(f), np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)
            # 求栅格中心坐标、在每个栅格上划分4或6个默认框。
            for i, j in zip(y, x):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 小正方形的边长
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                # 大正方形的边长
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                # 长方形的边长
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # 获得8732个默认框
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
