# Copyright (c) OpenMMLab. All rights reserved.
from pickle import TRUE
from numpy.core.records import fromfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import pdb
from ..losses import iou

import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel

from ..losses import occupancyloss

@SEGMENTORS.register_module()
class BEVSegmentor(BaseSegmentor):
    """Encoder Decoder segmentors for BEV perception.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 transformer=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BEVSegmentor, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        assert transformer is not None
        self.transformer = builder.build_neck(transformer)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, calib):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        x = self.transformer(x, calib)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x = self.extract_feat(img, calib)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        
        losses.update(add_prefix(loss_decode, 'decode'))
        
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # calib.shape [batch_size,3,3]
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x = self.extract_feat(img, calib)   
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            assert False
        seg_logit = self.whole_inference(img, img_meta, rescale)
        return seg_logit

    def simple_test(self, img, img_meta, rescale=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, False)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            assert False
        seg_logit = seg_logit.cpu().sigmoid()
  
        labels = torch.tensor(img_meta[0]['gt_semantic_seg'][None,...]).bool()        
        output_type = self.test_cfg.get('output_type','iou')
        positive_thred = self.test_cfg.get('positive_thred', 0.5)
        if output_type == 'iou':
            return [iou(seg_logit > positive_thred, labels[:,:-1,...], labels[:,-1,...], per_class=True),]
        elif output_type == 'seg':
            seg = seg_logit > positive_thred
            return [(seg.cpu().numpy(), labels.cpu().numpy(), img_meta[0]['filename']),]
        else:
            assert False,'unknown output type %s'% output_type
    
    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert False

@SEGMENTORS.register_module()
class origin_pyva_BEVSegmentor(BaseSegmentor):
    """Encoder Decoder segmentors for BEV perception.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                backbone,
                decode_head,
                neck=None,
                transformer=None,
                auxiliary_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                init_cfg=None):
        super(pyva_BEVSegmentor, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        assert transformer is not None
        self.transformer = builder.build_neck(transformer)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self,decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self,img,calib):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        output, transform_feature, retransform_features,forward_features = self.transformer(x,calib)
        return output,transform_feature,retransform_features,forward_features
    
    def encode_decode(self, img,img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x,_,_,forward_features= self.extract_feat(img,calib)
        out = self._decode_head_forward_test(x,forward_features,img_metas)
        return out

    def _decode_head_forward_train(self,x,img_metas,gt_semantic_seg,\
                    forward_features, transform_feature, retransform_features):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x,img_metas, gt_semantic_seg,forward_features, \
                                                transform_feature,retransform_features,self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    # def _decode_head_forward_test(self,x,transform_fature):
    def _decode_head_forward_test(self,inputs, forward_features,img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(inputs, forward_features,img_metas,self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                gt_semantic_seg,
                                                self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,img,img_metas,gt_semantic_seg):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x_feature,transform_feature,retransform_features,forward_features = self.extract_feat(img,calib)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_feature,img_metas,gt_semantic_seg,\
            forward_features, transform_feature, retransform_features)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x_feature, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide' or img_meta[0]['flip']:
            assert False
        seg_logit = self.whole_inference(img, img_meta, rescale)
        return seg_logit
    
    def simple_test(self, img, img_meta, rescale=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, False)
        if torch.onnx.is_in_onnx_export():
            assert False
        seg_logit = seg_logit.cpu().sigmoid()
        labels = torch.tensor(img_meta[0]['gt_semantic_seg'][None,...]).bool()
        # seg_logits shape:[1,14,196,200]
        # labels shape:[1,15,196,200]
        output_type = self.test_cfg.get('output_type','iou')
        positive_thred = self.test_cfg.get('positive_thred', 0.5)
        if output_type == 'iou':
            return [iou(seg_logit > positive_thred, labels[:,:-1,...], labels[:,-1,...], per_class=True),]
        elif output_type == 'seg':
            seg = seg_logit > positive_thred
            return [(seg.cpu().numpy(), labels.cpu().numpy(), img_meta[0]['filename']),]
        else:
            assert False,'unknown output type %s'% output_type

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        assert False

@SEGMENTORS.register_module()
class pyva_BEVSegmentor(BEVSegmentor):
    def __init__(self,**kwargs):
        super(pyva_BEVSegmentor, self).__init__(**kwargs)
        self.cycle_loss = nn.L1Loss()
        self.cycle_loss_weight = 0.001

    def extract_feat(self, img, calib):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        output, transform_feature, retransform_features,forward_features = self.transformer(x,calib)
        return output,transform_feature,retransform_features,forward_features
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x,_,_,_ = self.extract_feat(img, calib)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x_feature,_,retransform_features,forward_features = self.extract_feat(img,calib)
        losses = dict()
        losses['feature_loss'] = self.cycle_loss_weight * self.cycle_loss(retransform_features,forward_features).detach()
        losses_feature_loss = self.cycle_loss_weight * self.cycle_loss(retransform_features,forward_features)
        loss_decode = self._decode_head_forward_train(x_feature, img_metas,
                                                      gt_semantic_seg)
        # loss_decode is a dict, loss_decode has the following keys: decode.acc_seg,decode.loss_seg
        loss_decode['decode.loss_seg'] = loss_decode['decode.loss_seg'] + losses_feature_loss
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x_feature, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses


@SEGMENTORS.register_module()
class kd_pyva_BEVSegmentor(BEVSegmentor):
    def __init__(self,**kwargs):
        super(kd_pyva_BEVSegmentor, self).__init__(**kwargs)
        self.cycle_loss = nn.L1Loss()
        self.kd_loss = nn.MSELoss(reduction='mean')
        self.kd_loss_branch = nn.MSELoss(reduction='mean')
        self.cycle_loss_weight = 0.001
        self.kd_loss_weight_main = 0.002
        self.kd_loss_weight_branch = 0.001

    def extract_feat(self, img, calib):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        output,retransform_features,forward_features,notlearn_feats,learn_feats,notlearn_list,learn_list = self.transformer(x,calib)
        return output,retransform_features,forward_features,notlearn_feats,learn_feats,notlearn_list,learn_list
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x,_,_,_,_,_,_ = self.extract_feat(img, calib)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x_feature,retransform_features,forward_features,notlearn_feats,learn_feats,notlearn_list,learn_list = self.extract_feat(img,calib)
        losses = dict()
        losses['feature_loss'] = self.cycle_loss_weight * self.cycle_loss(retransform_features,forward_features).detach()
        losses_feature_loss = self.cycle_loss_weight * self.cycle_loss(retransform_features,forward_features)
        losses['kd_loss'] = self.kd_loss_weight_main * self.kd_loss(learn_feats,notlearn_feats).detach()
        losses_kd_loss = self.kd_loss_weight_main * self.kd_loss(learn_feats,notlearn_feats)
        losses['kd_loss_branch'] = self.kd_loss_weight_branch * sum([self.kd_loss_branch(learn_list[i],notlearn_list[i]) for i in range(5)]).detach()
        losses_kd_loss_branch = self.kd_loss_weight_branch * sum([self.kd_loss_branch(learn_list[i],notlearn_list[i]) for i in range(5)])
        loss_decode = self._decode_head_forward_train(x_feature, img_metas,gt_semantic_seg)
        # loss_decode is a dict, loss_decode has the following keys: decode.acc_seg,decode.loss_seg
        loss_decode['decode.loss_seg'] = loss_decode['decode.loss_seg'] + losses_feature_loss+losses_kd_loss+losses_kd_loss_branch
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x_feature, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, False)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            assert False
        seg_logit = seg_logit.cpu().sigmoid()
        positive_thred = 0.5
        labels = torch.tensor(img_meta[0]['gt_semantic_seg'][None,...]).bool() 
        output_type = self.test_cfg.get('output_type','iou')
        positive_thred = self.test_cfg.get('positive_thred', 0.5)
        if output_type == 'iou':
            return [iou(seg_logit>positive_thred, labels[:,:-1,...], labels[:,-1,...], per_class=True),]
        elif output_type == 'seg':
            seg = seg_logit > positive_thred
            return [(seg.cpu().numpy(), labels.cpu().numpy(), img_meta[0]['filename']),]
        else:
            assert False,'unknown output type %s'% output_type


@SEGMENTORS.register_module()
class kd_pon_vpn_BEVSegmentor(BEVSegmentor):
    def __init__(self,**kwargs):
        super(kd_pon_vpn_BEVSegmentor, self).__init__(**kwargs)
        self.kd_loss = nn.MSELoss(reduction='mean')
        self.kd_loss_branch = nn.MSELoss(reduction='mean')
        self.kd_loss_weight_main = 0.002
        self.kd_loss_weight_branch = 0.001

    def extract_feat(self, img, calib):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        output,notlearn_feats,learn_feats,notlearn_list,learn_list = self.transformer(x,calib)
        return output,notlearn_feats,learn_feats,notlearn_list,learn_list
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x,_,_,_,_ = self.extract_feat(img, calib)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x_feature,notlearn_feats,learn_feats,notlearn_list,learn_list = self.extract_feat(img,calib)
        losses = dict()
        losses['kd_loss'] = self.kd_loss_weight_main * self.kd_loss(learn_feats,notlearn_feats).detach()
        losses_kd_loss = self.kd_loss_weight_main * self.kd_loss(learn_feats,notlearn_feats)
        losses['kd_loss_branch'] = self.kd_loss_weight_branch * sum([self.kd_loss_branch(learn_list[i],notlearn_list[i]) for i in range(5)]).detach()
        losses_kd_loss_branch = self.kd_loss_weight_branch * sum([self.kd_loss_branch(learn_list[i],notlearn_list[i]) for i in range(5)])
        loss_decode = self._decode_head_forward_train(x_feature, img_metas,gt_semantic_seg)
        # loss_decode is a dict, loss_decode has the following keys: decode.acc_seg,decode.loss_seg
        loss_decode['decode.loss_seg'] = loss_decode['decode.loss_seg'] + losses_kd_loss+losses_kd_loss_branch
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x_feature, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, False)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            assert False
        seg_logit = seg_logit.cpu().sigmoid()
        positive_thred = 0.5
        labels = torch.tensor(img_meta[0]['gt_semantic_seg'][None,...]).bool() 
        output_type = self.test_cfg.get('output_type','iou')
        positive_thred = self.test_cfg.get('positive_thred', 0.5)
        if output_type == 'iou':
            return [iou(seg_logit, labels[:,:-1,...], labels[:,-1,...], per_class=True),]
        elif output_type == 'seg':
            seg = seg_logit > positive_thred
            return [(seg.cpu().numpy(), labels.cpu().numpy(), img_meta[0]['filename']),]
        else:
            assert False,'unknown output type %s'% output_type


@SEGMENTORS.register_module()
class force_lss_BEVSegmentor(BEVSegmentor):
    def __init__(self,**kwargs):
        super(force_lss_BEVSegmentor, self).__init__(**kwargs)
        self.force_loss = nn.L1Loss()
        self.force_loss_weight = 0.001

    def extract_feat(self, img, calib):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        output,learn_feats,notlearn_feats = self.transformer(x,calib)
        return output,learn_feats,notlearn_feats
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x,_,_ = self.extract_feat(img, calib)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        calib = torch.stack([data['calib'] for data in img_metas]).to(img)
        x_feature,learn_feats,notlearn_feats = self.extract_feat(img,calib)
        losses = dict()
        losses['learnable_loss'] = self.force_loss_weight * self.force_loss(learn_feats,notlearn_feats).detach()
        losses_learnable_loss = self.force_loss_weight * self.force_loss(learn_feats,notlearn_feats)
        loss_decode = self._decode_head_forward_train(x_feature, img_metas,
                                                      gt_semantic_seg)
        # loss_decode is a dict, loss_decode has the following keys: decode.acc_seg,decode.loss_seg
        loss_decode['decode.loss_seg'] = loss_decode['decode.loss_seg'] +losses_learnable_loss
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x_feature, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

