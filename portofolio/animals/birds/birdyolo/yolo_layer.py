#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math


__all__ = ["YoloLayer"]


class YoloLayer(nn.Module):
    def __init__(
        self,
        anchors_all,
        anchors_mask,
        num_classes,
        lambda_xy=1,
        lambda_wh=1,
        lambda_conf=1,
        lambda_cls=1,
        obj_scale=1,
        noobj_scale=1,
        ignore_thres=0.7,
        epsilon=1e-16,
    ):
        super(YoloLayer, self).__init__()

        assert num_classes > 0

        self._anchors_all = anchors_all
        self._anchors_mask = anchors_mask

        self._num_classes = num_classes
        self._bbox_attrib = 5 + num_classes

        self._lambda_xy = lambda_xy
        self._lambda_wh = lambda_wh
        self._lambda_conf = lambda_conf

        if self._num_classes == 1:
            self._lambda_cls = 0
        else:
            self._lambda_cls = lambda_cls

        self._obj_scale = obj_scale
        self._noobj_scale = noobj_scale
        self._ignore_thres = ignore_thres

        self._epsilon = epsilon

        self._mseloss = nn.MSELoss(reduction="sum")
        self._bceloss = nn.BCELoss(reduction="sum")
        self._bceloss_average = nn.BCELoss(reduction="elementwise_mean")

    def forward(self, x: torch.Tensor, img_dim: tuple, target=None):
        # x : batch_size * nA * (5 + num_classes) * H * W

        device = x.device
        if target is not None:
            assert target.device == x.device

        nB = x.shape[0]
        nA = len(self._anchors_mask)
        nH, nW = x.shape[2], x.shape[3]
        stride = img_dim[1] / nH
        anchors_all = torch.FloatTensor(self._anchors_all) / stride
        anchors = anchors_all[self._anchors_mask]

        # Reshape predictions from [B x [A * (5 + num_classes)] x H x W] to [B x A x H x W x (5 + num_classes)]
        preds = x.view(nB, nA, self._bbox_attrib, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

        # tx, ty, tw, wh
        preds_xy = preds[..., :2].sigmoid()
        preds_wh = preds[..., 2:4]
        preds_conf = preds[..., 4].sigmoid()
        preds_cls = preds[..., 5:].sigmoid()

        # calculate cx, cy, anchor mesh
        mesh_y, mesh_x = torch.meshgrid([torch.arange(nH, device=device), torch.arange(nW, device=device)])
        mesh_xy = torch.stack((mesh_x, mesh_y), 2).float()

        mesh_anchors = anchors.view(1, nA, 1, 1, 2).repeat(1, 1, nH, nW, 1).to(device)

        # pred_boxes holds bx,by,bw,bh
        pred_boxes = torch.FloatTensor(preds[..., :4].shape)
        pred_boxes[..., :2] = preds_xy + mesh_xy
        pred_boxes[..., 2:4] = preds_wh.exp() * mesh_anchors

        out = torch.cat((pred_boxes.to(device) * stride, preds_conf.to(device).unsqueeze(4), preds_cls.to(device),), 4,)

        # Reshape predictions from [B x A x H x W x (5 + num_classes)] to [B x [A x H x W] x (5 + num_classes)]
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(nB, nA * nH * nW, self._bbox_attrib)

        return out