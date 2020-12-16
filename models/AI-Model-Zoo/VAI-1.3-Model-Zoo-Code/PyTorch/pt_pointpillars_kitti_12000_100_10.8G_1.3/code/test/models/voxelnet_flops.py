# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import time
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import box_torch_ops
from models.pointpillars import PillarFeatureNet, PointPillarsScatter, RPN
from utils.torchprofiler import Profiler

import os
if os.environ["W_QUANT"]=='1':
    from pytorch_nndct.nn import QuantStub, DeQuantStub

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PreProcess(object):
    def __init__(self,
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self._with_distance = with_distance

        # scale x,y,z according to the range
        self.vz = voxel_size[2]
        self.pc_range = pc_range

    def __call__(self, features, num_voxels, coors):
        device = features.device
        '''
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # scale feature according to the range
        half_voxel_len = torch.Tensor([self.vx, self.vy, self.vz]).to(device) / 2.
        voxel_len = torch.Tensor([self.vx, self.vy, self.vz]).to(device)
        f_cluster = (f_cluster + half_voxel_len) / voxel_len
        f_center[:, :, 0] = (f_center[:, :, 0] + half_voxel_len[0]) / self.vx
        f_center[:, :, 1] = (f_center[:, :, 1] + half_voxel_len[1]) / self.vy
        '''
        pc_start = torch.from_numpy(self.pc_range[:3]).to(device)
        pc_len = torch.from_numpy(self.pc_range[3:6] - self.pc_range[:3]).to(device)
        features[:, :, :3] = features[:, :, :3] - pc_start
        features[:, :, :3] = features[:, :, :3] / pc_len

        # Combine together feature decorations
        # features_ls = [features, f_cluster, f_center]
        features_ls = [features]

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)
    
        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        features = features.permute(2, 0, 1).contiguous().unsqueeze(0)

        return features


class PostProcess(object):

    def __init__(self,
                 num_class=2,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None):
        super().__init__()
        self._box_coder = target_assigner.box_coder
        self._num_class = num_class
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_classifier = use_direction_classifier
        self._use_sigmoid_score = use_sigmoid_score
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._nms_score_threshold = nms_score_threshold

    def __call__(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)

        # self._total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds, batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                # print(dir_preds.shape)
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = torch.cat(selected_boxes, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_scores = torch.cat(selected_scores, dim=0)
                    if self._use_direction_classifier:
                        selected_dir_labels = torch.cat(
                            selected_dir_labels, dim=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self._nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    # opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
                # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
                # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
                # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
                # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="PillarFeatureNet",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="PointPillarsScatter",
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        # self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._dir_loss_ftor = None

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight

        if os.environ["W_QUANT"]=='1':
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        self.voxel_feature_extractor = PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=voxel_size,
            pc_range=pc_range
        )

        self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                            num_input_features=vfe_num_filters[-1])
        num_rpn_input_filters = self.middle_feature_extractor.nchannels

        self.rpn = RPN(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size)

        self.register_buffer("global_step", torch.LongTensor(1).zero_())

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, aug_voxels, coors):

        batch_size_dev = 1
        
        if os.environ["W_QUANT"]=='1':
            aug_voxels = self.quant(aug_voxels)
        voxel_features = self.voxel_feature_extractor(aug_voxels)
        if os.environ["W_QUANT"]=='1':
            voxel_features = self.dequant(voxel_features)

        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size_dev)

        if os.environ["W_QUANT"]=='1':
            spatial_features = self.quant(spatial_features)
        if self._use_bev:
            preds_dict = self.rpn(spatial_features, example["bev_map"])
        else:
            preds_dict = self.rpn(spatial_features)
        if os.environ["W_QUANT"]=='1':
            preds_dict = self.dequant(preds_dict)

        # (1)
        print("aug_voxels.shape: ", aug_voxels.shape)
        profiler = Profiler(self.voxel_feature_extractor.cpu())
        profiler.run_with_input(aug_voxels.cpu())
        profiler.print_summary()
        # (2)
        profiler = Profiler(self.middle_feature_extractor, device="cpu")
        profiler.run_with_input(voxel_features, coors, batch_size_dev)
        profiler.print_summary()
        # (3)
        print("spatial_features shape: ", spatial_features.shape)
        profiler = Profiler(self.rpn.cpu())
        one_spatial_features = spatial_features[0,...].unsqueeze(0)
        profiler.run_with_input(one_spatial_features.cpu())
        profiler.print_summary()
        print("Done")
        import sys
        sys.exit()

        return preds_dict


    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0


    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(net)
        return net

