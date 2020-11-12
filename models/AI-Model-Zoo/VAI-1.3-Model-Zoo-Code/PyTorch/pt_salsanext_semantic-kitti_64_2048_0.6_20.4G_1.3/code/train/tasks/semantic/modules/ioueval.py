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

#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import torch
from tasks.semantic.postproc.borderMask import borderMask


class iouEval:
    def __init__(self, n_classes, device, ignore=None):
        self.n_classes = n_classes
        self.device = device
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).double()
            #(self.n_classes, self.n_classes), device=self.device).float()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def addBatch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            #self.ones = torch.ones((idxs.shape[-1]), device=self.device).float()
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).double()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

        # print(self.tp.shape)
        # print(self.fp.shape)
        # print(self.fn.shape)

    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"


class biouEval(iouEval):
    def __init__(self, n_classes, device, ignore=None, border_size=1, kern_conn=4):
        super().__init__(n_classes, device, ignore)
        self.border_size = border_size
        self.kern_conn = kern_conn

        # check that I am only ignoring one class
        if len(ignore) > 1:
            raise ValueError("Length of ignored class list should be 1 or 0")
        elif len(ignore) == 0:
            ignore = None
        else:
            ignore = ignore[0]

        self.borderer = borderMask(self.n_classes, self.device,
                                   self.border_size, self.kern_conn,
                                   background_class=ignore)
        self.reset()

    def reset(self):
        super().reset()
        return

    def addBorderBatch1d(self, range_y, x, y, px, py):
        '''range_y=target as img, x=preds, y=targets, px,py=idxs of points of
           pointcloud in range img
           WARNING: Only batch size 1 works for now
        '''
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(range_y, np.ndarray):
            range_y = torch.from_numpy(np.array(range_y)).long().to(self.device)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)
        if isinstance(px, np.ndarray):
            px = torch.from_numpy(np.array(px)).long().to(self.device)
        if isinstance(py, np.ndarray):
            py = torch.from_numpy(np.array(py)).long().to(self.device)

        # get border mask of range_y
        border_mask_2d = self.borderer(range_y)

        # filter px, py according to if they are on border mask or not
        border_mask_1d = border_mask_2d[0, py, px].byte()

        # get proper points from filtered x and y
        x_in_mask = torch.masked_select(x, border_mask_1d)
        y_in_mask = torch.masked_select(y, border_mask_1d)

        # add batch
        self.addBatch(x_in_mask, y_in_mask)


if __name__ == "__main__":
    # mock problem
    nclasses = 2
    ignore = []

    # test with 2 squares and a known IOU
    lbl = torch.zeros((7, 7)).long()
    argmax = torch.zeros((7, 7)).long()

    # put squares
    lbl[2:4, 2:4] = 1
    argmax[3:5, 3:5] = 1

    # make evaluator
    eval = iouEval(nclasses, torch.device('cpu'), ignore)

    # run
    eval.addBatch(argmax, lbl)
    m_iou, iou = eval.getIoU()
    print("*" * 80)
    print("Small iou mock problem")
    print("IoU: ", m_iou)
    print("IoU class: ", iou)
    m_acc = eval.getacc()
    print("Acc: ", m_acc)
    print("*" * 80)
