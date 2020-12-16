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
from pathlib import Path

import numba
import numpy as np
from numba import cuda

from utils.buildtools.pybind11_build import load_pb11

try:
    from .nms import non_max_suppression
except:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        ["../cc/nms/nms_kernel.cu.cc", "../cc/nms/nms.cc"],
        current_dir / "nms.so",
        current_dir,
        cuda=True)
    from .nms import non_max_suppression


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def iou_device(a, b):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    width = max(right - left + 1, 0.)
    height = max(bottom - top + 1, 0.)
    interS = width * height
    Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return interS / (Sa + Sb - interS)


@cuda.jit('(int64, float32, float32[:, :], uint64[:])')
def nms_kernel_v2(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(
        shape=(threadsPerBlock, 5), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx, 0] = dev_boxes[dev_box_idx, 0]
        block_boxes[tx, 1] = dev_boxes[dev_box_idx, 1]
        block_boxes[tx, 2] = dev_boxes[dev_box_idx, 2]
        block_boxes[tx, 3] = dev_boxes[dev_box_idx, 3]
        block_boxes[tx, 4] = dev_boxes[dev_box_idx, 4]
    cuda.syncthreads()
    if (cuda.threadIdx.x < row_size):
        cur_box_idx = threadsPerBlock * row_start + cuda.threadIdx.x
        # cur_box = dev_boxes + cur_box_idx * 5;
        i = 0
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            if (iou_device(dev_boxes[cur_box_idx], block_boxes[i]) >
                    nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t


@cuda.jit('(int64, float32, float32[:], uint64[:])')
def nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if (tx < row_size):
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            iou = iou_device(dev_boxes[cur_box_idx * 5:cur_box_idx * 5 + 4],
                             block_boxes[i * 5:i * 5 + 4])
            if (iou > nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba.jit(nopython=True)
def nms_postprocess(keep_out, mask_host, boxes_num):
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    remv = np.zeros((col_blocks), dtype=np.uint64)
    num_to_keep = 0
    for i in range(boxes_num):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock
        mask = np.array(1 << inblock, dtype=np.uint64)
        if not (remv[nblock] & mask):
            keep_out[num_to_keep] = i
            num_to_keep += 1
            # unsigned long long *p = &mask_host[0] + i * col_blocks;
            for j in range(nblock, col_blocks):
                remv[j] |= mask_host[i * col_blocks + j]
                # remv[j] |= p[j];
    return num_to_keep


def nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. 
    
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """

    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    mask_host = np.zeros((boxes_num * col_blocks, ), dtype=np.uint64)
    blockspergrid = (div_up(boxes_num, threadsPerBlock),
                     div_up(boxes_num, threadsPerBlock))
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    # stream.synchronize()
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


def nms_gpu_cc(dets, nms_overlap_thresh, device_id=0):
    boxes_num = dets.shape[0]
    keep = np.zeros(boxes_num, dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    sorted_dets = dets[order, :]
    num_out = non_max_suppression(sorted_dets, keep, nms_overlap_thresh,
                                  device_id)
    keep = keep[:num_out]
    return list(order[keep])


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    return (
        (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2, ), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2, ), dtype=numba.float32)
        vs = cuda.local.array((16, ), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2, ), dtype=numba.float32)
    B = cuda.local.array((2, ), dtype=numba.float32)
    C = cuda.local.array((2, ), dtype=numba.float32)
    D = cuda.local.array((2, ), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2, ), dtype=numba.float32)
    b = cuda.local.array((2, ), dtype=numba.float32)
    c = cuda.local.array((2, ), dtype=numba.float32)
    d = cuda.local.array((2, ), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2, ), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4, ), dtype=numba.float32)
    corners_y = cuda.local.array((4, ), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i +
                1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8, ), dtype=numba.float32)
    corners2 = cuda.local.array((8, ), dtype=numba.float32)
    intersection_corners = cuda.local.array((16, ), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def devRotateIoU(rbox1, rbox2):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    return area_inter / (area1 + area2 - area_inter)


@cuda.jit('(int64, float32, float32[:], uint64[:])')
def rotate_nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 6, ), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if (tx < col_size):
        block_boxes[tx * 6 + 0] = dev_boxes[dev_box_idx * 6 + 0]
        block_boxes[tx * 6 + 1] = dev_boxes[dev_box_idx * 6 + 1]
        block_boxes[tx * 6 + 2] = dev_boxes[dev_box_idx * 6 + 2]
        block_boxes[tx * 6 + 3] = dev_boxes[dev_box_idx * 6 + 3]
        block_boxes[tx * 6 + 4] = dev_boxes[dev_box_idx * 6 + 4]
        block_boxes[tx * 6 + 5] = dev_boxes[dev_box_idx * 6 + 5]
    cuda.syncthreads()
    if (tx < row_size):
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if (row_start == col_start):
            start = tx + 1
        for i in range(start, col_size):
            iou = devRotateIoU(dev_boxes[cur_box_idx * 6:cur_box_idx * 6 + 5],
                               block_boxes[i * 6:i * 6 + 5])
            # print('iou', iou, cur_box_idx, i)
            if (iou > nms_overlap_thresh):
                t |= 1 << i
        col_blocks = ((n_boxes) // (threadsPerBlock) + (
            (n_boxes) % (threadsPerBlock) > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t


def rotate_nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. WARNING: this function can provide right result 
    but its performance isn't be tested
    
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    dets = dets.astype(np.float32)
    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    # mask_host shape: boxes_num * col_blocks * sizeof(np.uint64)
    mask_host = np.zeros((boxes_num * col_blocks, ), dtype=np.uint64)
    blockspergrid = (div_up(boxes_num, threadsPerBlock),
                     div_up(boxes_num, threadsPerBlock))
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        rotate_nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


@cuda.jit('(int64, int64, float32[:], float32[:], float32[:])', fastmath=False)
def rotate_iou_kernel(N, K, dev_boxes, dev_query_boxes, dev_iou):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoU(block_qboxes[i * 5:i * 5 + 5],
                                           block_boxes[tx * 5:tx * 5 + 5])


def rotate_iou_gpu(boxes, query_boxes, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit(
    '(int64, int64, float32[:], float32[:], float32[:], int32)',
    fastmath=False)
def rotate_iou_kernel_eval(N,
                           K,
                           dev_boxes,
                           dev_query_boxes,
                           dev_iou,
                           criterion=-1):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 5:i * 5 + 5],
                                               block_boxes[tx * 5:tx * 5 + 5],
                                               criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)
