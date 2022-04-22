import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS

def feature_selection(input, dim, index):
    # feature selection
    # input: [N, ?, ?, ...]
    # dim: scalar > 0
    # index: [N, idx]
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
    expanse = list(input.size())
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

class CrossViewTransformer(nn.Module):
    def __init__(self, in_dim=128):
        super(CrossViewTransformer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.f_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=3, stride=1, padding=1,
                                bias=True)

    def forward(self, front_x, cross_x, front_x_hat):
        m_batchsize, C, width, height = front_x.size()
        proj_query = self.query_conv(cross_x).view(m_batchsize, -1, width * height)  # B x C x (N)
        proj_key = self.key_conv(front_x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x C x (W*H)

        energy = torch.bmm(proj_key, proj_query)  # transpose check
        front_star, front_star_arg = torch.max(energy, dim=1)
        proj_value = self.value_conv(front_x_hat).view(m_batchsize, -1, width * height)  # B x C x N

        T = feature_selection(proj_value, 2, front_star_arg).view(front_star.size(0), -1, width, height)

        S = front_star.view(front_star.size(0), 1, width, height)
        # according to github issue,front_x should be cross_x, https://github.com/JonDoe-297/cross-view/issues/4
        front_res = torch.cat((cross_x, T), dim=1)
        front_res = self.f_conv(front_res)
        front_res = front_res * S
        # according to github issue,front_x should be cross_x, https://github.com/JonDoe-297/cross-view/issues/4
        # output = front_x + front_res
        output = cross_x + front_res

        return output

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class CycledViewProjection(nn.Module):
    def __init__(self, in_dim=8):
        super(CycledViewProjection, self).__init__()
        self.transform_module = TransformModule(dim=in_dim)
        self.retransform_module = TransformModule(dim=in_dim)

    def forward(self, x):
        B, C, H, W = x.view([-1, int(x.size()[1])] + list(x.size()[2:])).size()
        transform_feature = self.transform_module(x)
        transform_features = transform_feature.view([B, int(x.size()[1])] + list(x.size()[2:]))
        retransform_features = self.retransform_module(transform_features)
        return transform_feature, retransform_features


class TransformModule(nn.Module):
    def __init__(self, dim=8, use_heavy=False):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.use_heavy = use_heavy
        self.mat_list = nn.ModuleList()
        if not self.use_heavy:
            self.fc_transform = nn.Sequential(
                nn.Linear(dim * dim, dim * dim),
                nn.ReLU(),
                nn.Linear(dim * dim, dim * dim),
                nn.ReLU()
            )
        else:
            self.fc_transform = nn.Sequential(
                nn.Linear(dim * dim, dim * dim * 784),
                nn.ReLU(),
                nn.Linear(dim * dim * 784, dim * dim),
                nn.ReLU()
            )
    def forward(self, x):
        # shape x: B, C, H, W
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim])
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb

class Interpo(nn.Module):
    def __init__(self,size):
        super(Interpo,self).__init__()
        self.size = size

    def forward(self,x):
        x = F.interpolate(x,size = self.size,mode = 'bilinear',align_corners=True)
        return x

@NECKS.register_module()
class Pyva_transformer(nn.Module):
    def __init__(self,size,back = 'swin',use_fpn=False,use_heavy=False):
        super(Pyva_transformer,self).__init__()
        self.size = size
        self.back = back
        self.use_fpn = use_fpn
        self.use_heavy = use_heavy
        if self.back=='res18':
            self.conv1 = Conv3x3(512,512)
        if self.back=='res18' and self.use_fpn: 
            self.conv1 = Conv3x3(256,512)
        if self.back=='res50':
            self.conv1 = Conv3x3(2048,512)
        if self.back=='swin':
            self.conv1 = Conv3x3(768,512)
        self.conv2 = Conv3x3(512,128)
        self.pool = nn.MaxPool2d(2)
        self.transform_feature = TransformModule(use_heavy=self.use_heavy,dim=8)
        self.retransform_feature = TransformModule(use_heavy=self.use_heavy,dim=8)
        self.crossview = CrossViewTransformer(in_dim = 128)
        self.interpolate = Interpo(self.size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv5 = Conv3x3(64,64)

    def forward(self,x,calib):
        x = x[-1]  # bs,2048, 38, 50
        # if we use swin as backbone, x.shape = [bs,768,32,32];if we use res18 as backbone, x.shape: [bs,512,32,32]
        x = self.interpolate(x) # bs,2048, 32, 32
        x = self.conv1(x)  # bs,512, 32, 32
        x = self.pool(x)   # bs,512, 16, 16
        x = self.conv2(x)  # bs,128, 16, 16
        x = self.pool(x)   # bs,128, 8, 8
        B,C,H,W = x.shape
        transform_feature = self.transform_feature(x)  # bs,128, 8, 8
        retransform_feature = self.retransform_feature(transform_feature)  # bs,128, 8, 8
        feature_final = self.crossview(x.view(B,C,H,W),
                                    transform_feature.view(B,C,H,W),
                                    retransform_feature.view(B,C,H,W))     # bs,128, 8, 8
        feature_final = F.interpolate(feature_final,scale_factor=2,mode='nearest')
        feature_final = self.conv3(feature_final)  # bs,128, 16,16
        feature_final = F.interpolate(feature_final,scale_factor=2,mode='nearest')
        feature_final = self.conv4(feature_final)  # bs,64, 32, 32

        feature_final = F.interpolate(
            feature_final,
            size = (98,100),
            mode = 'bilinear',
            align_corners = True
        )  # bs,64, 98,100
        feature_final = self.conv5(feature_final)
        return feature_final,x,retransform_feature,transform_feature

