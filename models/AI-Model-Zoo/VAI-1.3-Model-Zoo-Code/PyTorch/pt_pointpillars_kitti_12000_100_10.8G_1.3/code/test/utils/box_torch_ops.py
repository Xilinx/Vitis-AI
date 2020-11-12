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


import math
from functools import reduce

import numpy as np
import torch
from torch import FloatTensor as FTensor
from torch import stack as tstack

# import torchplus
from utils.box_np_ops import iou_jit
from utils.non_max_suppression.nms_gpu import (nms_gpu, rotate_iou_gpu, rotate_nms_gpu)
from utils.non_max_suppression.nms_cpu import rotate_nms_cc


def torch_to_np_dtype(ttype):
    type_map = { 
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }   
    return type_map[ttype]


def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    za = za + ha / 2
    zg = zg + hg / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

    # rt = rg - ra
    # return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)


def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)

    else:
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, wa, la, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, wg, lg, rg = torch.split(boxes, 1, dim=-1)
    diagonal = torch.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, wt, lt, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, wt, lt, rt], dim=-1)

    # rt = rg - ra
    # return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)


def bev_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, wa, la, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)

    else:
        xt, yt, wt, lt, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    return torch.cat([xg, yg, wg, lg, rg], dim=-1)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners


def corners_2d(dims, origin=0.5):
    """generate relative 2d box corners based on length per dim and
    origin point.
    
    Args:
        dims (float array, shape=[N, 2]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
        float array, shape=[N, 4, 2]: returned corners. 
        point layout: x0y0, x0y1, x1y1, x1y0
    """
    return corners_nd(dims, origin)


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = tstack([
            tstack([rot_cos, zeros, -rot_sin]),
            tstack([zeros, ones, zeros]),
            tstack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tstack([
            tstack([rot_cos, -rot_sin, zeros]),
            tstack([rot_sin, rot_cos, zeros]),
            tstack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tstack([
            tstack([zeros, rot_cos, -rot_sin]),
            tstack([zeros, rot_sin, rot_cos]),
            tstack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = math.sin(angle)
    rot_cos = math.cos(angle)
    point_type = torchplus.get_tensor_class(points)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.tensor([rot_cos, 0, -rot_sin], dtype=points.dtype, device=points.device),
            torch.tensor([0, 1, 0], dtype=points.dtype, device=points.device),
            torch.tensor([rot_sin, 0, rot_cos], dtype=points.dtype, device=points.device)
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.tensor([rot_cos, -rot_sin, 0], dtype=points.dtype, device=points.device),
            torch.tensor([rot_sin, rot_cos, 0], dtype=points.dtype, device=points.device),
            torch.tensor([0, 0, 1], dtype=points.dtype, device=points.device)
        ])
    elif axis == 0:
        rot_mat_T = torch.stack([
            torch.tensor([1, 0, 0], dtype=points.dtype, device=points.device),
            torch.tensor([0, rot_cos, -rot_sin], dtype=points.dtype, device=points.device),
            torch.tensor([0, rot_sin, rot_cos], dtype=points.dtype, device=points.device)
        ])
    else:
        raise ValueError("axis should in range")
    return points @ rot_mat_T


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack(
        [tstack([rot_cos, -rot_sin]),
         tstack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def project_to_image(points_3d, proj_mat):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = torch.cat(
        [points_3d, torch.zeros(*points_shape).type_as(points_3d)], dim=-1)
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat(
        [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    lidar_points = points @ torch.inverse((r_rect @ velo2cam).t())
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    num_points = points.shape[0]
    points = torch.cat(
        [points, torch.ones(num_points, 1).type_as(points)], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[..., 0:3]
    l, h, w = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return torch.cat([xyz_lidar, w, l, h, r], dim=-1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def multiclass_nms(nms_func,
                   boxes,
                   scores,
                   num_class,
                   pre_max_size=None,
                   post_max_size=None,
                   score_thresh=0.0,
                   iou_threshold=0.5):
    # only output [selected] * num_class, please slice by your self
    selected_per_class = []
    assert len(boxes.shape) == 3, "bbox must have shape [N, num_cls, 7]"
    assert len(scores.shape) == 2, "score must have shape [N, num_cls]"
    num_class = scores.shape[1]
    if not (boxes.shape[1] == scores.shape[1] or boxes.shape[1] == 1):
        raise ValueError('second dimension of boxes must be either 1 or equal '
                         'to the second dimension of scores')
    num_boxes = boxes.shape[0]
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]
    boxes_ids = (range(num_classes)
                 if boxes.shape[1] > 1 else [0] * num_classes)
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
        # for class_idx in range(1, num_class):
        class_scores = scores[:, class_idx]
        class_boxes = boxes[:, boxes_idx]
        if score_thresh > 0.0:
            class_scores_keep = torch.nonzero(class_scores >= score_thresh)
            if class_scores_keep.shape[0] != 0:
                class_scores_keep = class_scores_keep[:, 0]
            else:
                selected_per_class.append(None)
                continue
            class_scores = class_scores[class_scores_keep]
        if class_scores.shape[0] != 0:
            if score_thresh > 0.0:
                class_boxes = class_boxes[class_scores_keep]
            keep = nms_func(class_boxes, class_scores, pre_max_size,
                            post_max_size, iou_threshold)
            if keep is not None:
                if score_thresh > 0.0:
                    selected_per_class.append(class_scores_keep[keep])
                else:
                    selected_per_class.append(keep)
            else:
                selected_per_class.append(None)
        else:
            selected_per_class.append(None)
    return selected_per_class

def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    dets = torch.cat([bboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().cuda()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().cuda()


def rotate_nms(rbboxes,
               scores,
               pre_max_size=None,
               post_max_size=None,
               iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        rbboxes = rbboxes[indices]
    dets = torch.cat([rbboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(rotate_nms_cc(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().cuda()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().cuda()
