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


import numpy as np


def create_anchors_3d_stride(feature_size,
                             sizes=[1.6, 3.9, 1.56],
                             anchor_strides=[0.4, 0.4, 0.0],
                             anchor_offsets=[0.2, -39.8, -1.78],
                             rotations=[0, np.pi / 2], 
                             dtype=np.float32):
    """ 
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=dtype)
    y_centers = np.arange(feature_size[1], dtype=dtype)
    x_centers = np.arange(feature_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3]) 
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5 
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3]) 
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1 
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5]) 


def create_anchors_3d_range(feature_size,
                            anchor_range,
                            sizes=[1.6, 3.9, 1.56],
                            rotations=[0, np.pi / 2],
                            dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


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
        return create_anchors_3d_stride(
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
        return create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._dtype)



def build(anchor_config):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    ag_type = anchor_config.WhichOneof('anchor_generator')

    if ag_type == 'anchor_generator_stride':
        config = anchor_config.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    elif ag_type == 'anchor_generator_range':
        config = anchor_config.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    else:
        raise ValueError(" unknown anchor generator type")
