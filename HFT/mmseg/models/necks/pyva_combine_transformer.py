import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS

import pdb
import math
from mmcv.runner import BaseModule
from functools import reduce
from operator import mul

# generate grids in BEV
def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1)


class Resampler(nn.Module):

    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        # extents[1]:zmin,extents[3]:zmax
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)

    def forward(self, features, calib):
        # Copy grid to the correct device
        self.grid = self.grid.to(features)

        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        # calib shape:[bs,3,3]-->[bs,2,3]-->[bs,2,2]-->[bs,1,1,2,2]
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., 1]
        ucoords = ucoords / features.size(-1) * 2 - 1

        # Normalize z coordinates
        zcoords = (cam_coords[..., 1] - self.near) / (self.far - self.near) * 2 - 1

        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


class DenseTransformer(nn.Module):

    def __init__(self, in_channels, channels, resolution, grid_extents,
                 ymin, ymax, focal_length, groups=1):
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)
        self.ymid = (ymin + ymax) / 2

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            channels * self.in_height, channels * self.out_depth, 1, groups=groups
        )
        self.out_channels = channels

    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        features = torch.stack([self._crop_feature_map(fmap, cal)
                                for fmap, cal in zip(features, calib)])

        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Flatten height and channel dimensions
        B, C, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        bev_feats = self.fc(flat_feats).view(B, C, -1, W)

        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)

    def _crop_feature_map(self, fmap, calib):
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])

def feature_selection(input, dim, index):
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
    def __init__(self, dim=8):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.mat_list = nn.ModuleList()
        self.fc_transform = nn.Sequential(
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU()
        )
    def forward(self, x):
        # shape x: B, C, H, W
        # x = self.bn(x)
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim])
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb

@NECKS.register_module()
class Pyva_combine_transformer(BaseModule):
    def __init__(self,size,back = 'swin',in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0):
        super(Pyva_combine_transformer,self).__init__()
        self.size = size
        self.back = back
        # pyramid transformer
        self.transformers = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                   subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
        
        if self.back=='res50':
            self.conv1 = Conv3x3(2048,512)
        if self.back=='swin':
            self.conv1 = Conv3x3(256,512)
        self.conv2 = Conv3x3(512,128)
        self.pool = nn.MaxPool2d(2)
        self.transform_feature = TransformModule(dim=8)
        self.retransform_feature = TransformModule(dim=8)
        self.crossview = CrossViewTransformer(in_dim = 128)
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
        feature_maps = x
        x = x[2]  # bs,256,32,32
        # if we use swin as backbone, x.shape = [bs,256,32,32]
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

        bev_feats = list()
        # scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))

        bev_feats = torch.cat(bev_feats[::-1], dim=-2)

        feature_final = F.interpolate(
            feature_final,
            size = (98,100),
            mode = 'bilinear',
            align_corners = True
        )  # bs,64, 98,100
        feature_final = self.conv5(feature_final)

        # combine learnable and unlearnable feature
        feature_final_out = feature_final+bev_feats
        return feature_final_out,x,retransform_feature,transform_feature

@NECKS.register_module()
class Pon_combine_vpn_simple_fpn_force_transformer(BaseModule):
    def __init__(self,size,back = 'swin',use_light=False, in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                 extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0,
                 input_width=32, input_height=32, input_dim=256, output_width=100, output_height=98, output_dim=64):
        super(Pon_combine_vpn_simple_fpn_force_transformer,self).__init__()
        self.size = size
        self.back = back
        self.use_light = use_light
        self.depth_list = [7,10,20,39,22]
        # pyramid transformer
        self.transformers = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)
            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                   subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
        
        self.conv5 = Conv3x3(64,64)
        self.conv_down_1 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)
        self.conv_down_2 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)

        # vpn init part
        self.input_width=input_width
        self.input_height=input_height
        self.input_dim=input_dim
        self.output_width=output_width
        self.output_height=output_height
        self.output_dim=output_dim
        if self.use_light:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.05*self.output_width * self.output_height)),
                nn.ReLU(),
                nn.Linear(int(0.05*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        else:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, self.output_width * self.output_height),
                nn.ReLU(),
                nn.Linear(self.output_width * self.output_height, self.output_width * self.output_height),
                nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(self.input_dim, self.output_dim, 1),
                                  nn.ReLU())

    def forward(self,x,calib):
        feature_maps = x
        x_0,x_1,x_2,x_3,x_4 = x
        x_1 = x_1+self.conv_down_1(x_0)
        x_2 = x_2+self.conv_down_2(x_1)
        x_3 = x_3+F.interpolate(x_4,scale_factor=2,mode='bilinear',align_corners=True)
        x_2 = x_2+F.interpolate(x_3,scale_factor=2,mode='bilinear',align_corners=True)
        x = x_2  # bs,256,32,32

        # learnable part
        n,c,h,w = x.shape
        x = x.view(n, c, h*w)
        x = self.tf(x)
        x = x.view(n, c, self.output_height, self.output_width)
        feature_final = self.conv(x)

        feature_final_1 = F.interpolate(feature_final,size = (self.depth_list[0],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_2 = F.interpolate(feature_final,size = (self.depth_list[1],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_3 = F.interpolate(feature_final,size = (self.depth_list[2],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_4 = F.interpolate(feature_final,size = (self.depth_list[3],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_5 = F.interpolate(feature_final,size = (self.depth_list[4],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_list = [feature_final_1,feature_final_2,feature_final_3,feature_final_4,feature_final_5]
        feature_final = torch.cat(feature_final_list[::-1], dim=-2)
        feature_final = self.conv5(feature_final)

        # unlearnable part
        bev_feats = list()
        # scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))

        bev_feats_out = torch.cat(bev_feats[::-1], dim=-2)

        # combine learnable and unlearnable feature
        feature_final_out = feature_final+bev_feats_out
        return feature_final_out,bev_feats_out,feature_final,bev_feats[::-1],feature_final_list

@NECKS.register_module()
class Pyva_combine_simple_force_transformer(BaseModule):
    def __init__(self,size,back = 'swin',in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                 extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0):
        super(Pyva_combine_simple_force_transformer,self).__init__()

        self.size = size
        self.back = back
        # pyramid transformer
        self.transformers = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                   subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
        
        if self.back=='res50':
            self.conv1 = Conv3x3(2048,512)
        if self.back=='swin':
            self.conv1 = Conv3x3(256,512)
        self.conv2 = Conv3x3(512,128)
        self.pool = nn.MaxPool2d(2)
        self.transform_feature = TransformModule(dim=8)
        self.retransform_feature = TransformModule(dim=8)
        self.crossview = CrossViewTransformer(in_dim = 128)
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
        self.conv_down_1 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)
        self.conv_down_2 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)
    def forward(self,x,calib):
        feature_maps = x
        # x[0].shape: bs,256,128,128   x[1].shape: bs,256,64,64
        # x[2].shape: bs,256,32,32     x[3].shape: bs,256,16,16
        x_0,x_1,x_2,x_3,x_4 = x
        x_1 = x_1+self.conv_down_1(x_0)
        x_2 = x_2+self.conv_down_2(x_1)
        x_3 = x_3+F.interpolate(x_4,scale_factor=2,mode='bilinear',align_corners=True)
        x_2 = x_2+F.interpolate(x_3,scale_factor=2,mode='bilinear',align_corners=True)
        x = x_2  # bs,256,32,32
        # if we use swin as backbone, x.shape = [bs,256,32,32]
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

        bev_feats = list()
        # scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))

        bev_feats_out = torch.cat(bev_feats[::-1], dim=-2)

        feature_final = F.interpolate(
            feature_final,
            size = (98,100),
            mode = 'bilinear',
            align_corners = True
        )  # bs,64, 98,100
        feature_final = self.conv5(feature_final)
        feature_final_list = [feature_final[:,:,:7,:],feature_final[:,:,7:17,:],feature_final[:,:,17:37,:],feature_final[:,:,37:76,:],feature_final[:,:,76:,:]]
        # combine learnable and unlearnable feature
        feature_final_out = feature_final+bev_feats_out
        return feature_final_out,retransform_feature,transform_feature,bev_feats_out,feature_final,bev_feats[::-1],feature_final_list

@NECKS.register_module()
class Pyva_combine_simple_fpn_force_transformer(BaseModule):
    def __init__(self,size,back = 'swin',in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                 extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0):
        super(Pyva_combine_simple_fpn_force_transformer,self).__init__()
        self.size = size
        self.back = back
        self.depth_list = [7,10,20,39,22]
        # pyramid transformer
        self.transformers = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)
            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                   subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
        
        if self.back=='res50':
            self.conv1 = Conv3x3(2048,512)
        if self.back=='swin':
            self.conv1 = Conv3x3(256,512)
        self.conv2 = Conv3x3(512,128)
        self.pool = nn.MaxPool2d(2)
        self.transform_feature = TransformModule(dim=8)
        self.retransform_feature = TransformModule(dim=8)
        self.crossview = CrossViewTransformer(in_dim = 128)
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
        self.conv_down_1 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)
        self.conv_down_2 = nn.Conv2d(256,256,kernel_size=3,padding=1,stride=2)
    def forward(self,x,calib):
        feature_maps = x
        x_0,x_1,x_2,x_3,x_4 = x
        x_1 = x_1+self.conv_down_1(x_0)
        x_2 = x_2+self.conv_down_2(x_1)
        x_3 = x_3+F.interpolate(x_4,scale_factor=2,mode='bilinear',align_corners=True)
        x_2 = x_2+F.interpolate(x_3,scale_factor=2,mode='bilinear',align_corners=True)
        x = x_2  # bs,256,32,32
        # if we use swin as backbone, x.shape = [bs,256,32,32]
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

        feature_final_1 = F.interpolate(feature_final,size = (self.depth_list[0],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_2 = F.interpolate(feature_final,size = (self.depth_list[1],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_3 = F.interpolate(feature_final,size = (self.depth_list[2],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_4 = F.interpolate(feature_final,size = (self.depth_list[3],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_5 = F.interpolate(feature_final,size = (self.depth_list[4],100),mode = 'bilinear',align_corners = True)  # bs,64, 22,100
        feature_final_list = [feature_final_1,feature_final_2,feature_final_3,feature_final_4,feature_final_5]
        feature_final = torch.cat(feature_final_list[::-1], dim=-2)
        feature_final = self.conv5(feature_final)
        
        bev_feats = list()
        # scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))

        # Combine birds-eye-view feature maps along the depth axis
        # bev_feats.shape:[bs,64,98,100]
        bev_feats_out = torch.cat(bev_feats[::-1], dim=-2)

        # combine learnable and unlearnable feature
        feature_final_out = feature_final+bev_feats_out
        return feature_final_out,retransform_feature,transform_feature,bev_feats_out,feature_final,bev_feats[::-1],feature_final_list

