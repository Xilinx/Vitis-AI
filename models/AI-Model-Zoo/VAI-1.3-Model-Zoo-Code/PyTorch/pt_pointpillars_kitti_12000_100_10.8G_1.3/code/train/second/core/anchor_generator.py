
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

import numpy as np
from second.core import box_np_ops


class AnchorGeneratorStride:
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 class_id=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._dtype = dtype
        self._class_id = class_id
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_id(self):
        return self._class_id

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        return box_np_ops.create_anchors_3d_stride(
            feature_map_size, self._sizes, self._anchor_strides,
            self._anchor_offsets, self._rotations, self._dtype)

class AnchorGeneratorRange:
    def __init__(self,
                 anchor_ranges,
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0, np.pi / 2],
                 class_id=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._class_id = class_id
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_id(self):
        return self._class_id

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        return box_np_ops.create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._dtype)
