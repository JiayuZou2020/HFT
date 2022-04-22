import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS
import pdb

@NECKS.register_module()
class TransformerLinear(BaseModule):
    def __init__(self, use_light=False, use_high_res=False, input_width=25, input_height=19, input_dim=768, output_width=100, output_height=98, output_dim=64):
        super(TransformerLinear, self).__init__()
        self.use_light = use_light
        self.use_hight_res = use_high_res
        if not use_high_res:
            self.input_width=input_width
            self.input_height=input_height
        else:
            self.input_width=32
            self.input_height=32
        
        self.input_dim=input_dim
        self.output_width=output_width
        self.output_height=output_height
        self.output_dim=output_dim
        if not self.use_light:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.2*(self.output_width * self.output_height))),
                nn.ReLU(),
                nn.Linear(int(0.2*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        else:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.05*self.output_width * self.output_height)),
                nn.ReLU(),
                nn.Linear(int(0.05*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim, self.output_dim, 1),
                                  nn.ReLU())

    def forward(self, feature_maps, intrinstics):
        feature = feature_maps[3]
        n,c,h,w = feature.shape
        feature = feature.view(n, c, h*w)
        feature = self.tf(feature)
        feature = feature.view(n, c, self.output_height, self.output_width)
        feature = self.conv(feature)
        return feature
