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

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

class Criterion(nn.Module):
    def __init__(self, ignore_index=255, weight=None, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        class_wts = torch.ones(len(weight))
        for i in range(len(weight)):
            class_wts[i] = weight[i]
        print(class_wts)
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_wts, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss

