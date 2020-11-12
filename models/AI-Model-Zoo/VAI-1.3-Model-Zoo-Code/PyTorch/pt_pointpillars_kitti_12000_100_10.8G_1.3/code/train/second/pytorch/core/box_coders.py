
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

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from second.core.box_coders import GroundBox3dCoder, BevBoxCoder
from second.pytorch.core import box_torch_ops
import torch

class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors, self.vec_encode, self.linear_dim)



class BevBoxCoderTorch(BevBoxCoder):
    def encode_torch(self, boxes, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        boxes = boxes[..., [0, 1, 3, 4, 6]]
        return box_torch_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_torch(self, encodings, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = box_torch_ops.bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
        z_fixed = torch.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype, device=ret.device)
        h_fixed = torch.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype, device=ret.device)
        return torch.cat([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], dim=-1)


