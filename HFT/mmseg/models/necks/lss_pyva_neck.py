import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from functools import reduce
from operator import mul
from ..builder import NECKS

# lss part
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)
        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None

# pyva part
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
        # self.ymid = 1
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
        # H is not fixed every time
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


@NECKS.register_module()
class LSS_PYVA_neck(BaseModule):
    def __init__(self, in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                 extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0,
                 downsample=32, bev_feature_channels=64, ogfH=1024, ogfW=1024,
                 grid_conf=dict(dbound=[1, 50, 1], xbound=[-25,25,0.5], zbound=[1, 50, 0.5], ybound=[-10,10,20])):
    # resolution = 0.25*(1*2)=0.5
    # focal = 78.75,39.375,19.6875,9.84375,4.921875
    # subset_extents = [-25,39,25,50],[-25,19.5,25,39],\
    # [-25,9.5,25,19.5],[-25,4.5,25,9.5],[-25,1,25,4.5]
    # ymin=-2,ymax=4
        super(LSS_PYVA_neck,self).__init__()

        # pyva part
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

        # lss part
        self.grid_conf = grid_conf
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.downsample = downsample
        self.ogfH = ogfH
        self.ogfW = ogfW
        self.frustum = self.create_frustum()
        self.D ,_,_,_ = self.frustum.shape
        self.C = bev_feature_channels
        # by default, self.C = 64, self.D = 49
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, kernel_size=1, padding=0)
        self.use_quickcumsum = True

    def create_frustum(self):
        fH, fW = self.ogfH//self.downsample, self.ogfW//self.downsample
        depth_samples = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        num_depth, _, _ = depth_samples.shape
        x_samples = torch.linspace(0, self.ogfW - 1, fW, dtype=torch.float).view(1,1,fW).expand(num_depth,fH,fW)
        y_samples = torch.linspace(0, self.ogfH - 1, fH, dtype=torch.float).view(1,fH,1).expand(num_depth,fH,fW)

        # D x H x W x 3
        frustum = torch.stack((x_samples,y_samples,depth_samples), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinstics):
        B = intrinstics.shape[0]
        D,H,W,C = self.frustum.shape
        points = self.frustum.view(1,D,H,W,-1).expand(B,D,H,W,C)
        points = torch.cat([points[:,:,:,:,:2]* points[:,:,:,:,2:3],
                            points[:,:,:,:,2:3]],4)
        combine = torch.inverse(intrinstics)
        points = combine.view(B,1,1,1,3,3).matmul(points.unsqueeze(-1)).squeeze(-1).view(B,1,D,H,W,-1)
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
                + geom_feats[:, 1] * (nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[1], nx[2], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 1], geom_feats[:, 2], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def forward(self, feature_maps, calib):
        geom = self.get_geometry(calib)  
        b,_,_,h,w,_  = geom.shape
        if self.downsample==32:
            # feature = feature_maps[3][:,:,:h,:w]  
            feature = feature_maps[2][:,:,:h,:w]  
        elif self.downsample==16:
            feature = feature_maps[0][:, :, :h, :w]
        else:
            assert False
        feature = self.depthnet(feature)   
        depth = self.get_depth_dist(feature[:, :self.D])     
        new_feature = depth.unsqueeze(1) * feature[:, self.D:(self.D + self.C)].unsqueeze(2)  
        feature = new_feature.view(b, 1, self.C, self.D, h, w)  
        feature = feature.permute(0, 1, 3, 4, 5, 2)  
        # feature_lss.shape: bs,64,98,100
        feature_lss =  self.voxel_pooling(geom, feature)

        # pyva part
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
        feature_pyva = torch.cat(bev_feats[::-1], dim=-2)
        feature_final  = feature_pyva+feature_lss
        return feature_final,feature_pyva,feature_lss