
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

from second.core.voxel_generator import VoxelGenerator
from second.protos import voxel_generator_pb2


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_config.voxel_size),
        point_cloud_range=list(voxel_config.point_cloud_range),
        max_num_points=voxel_config.max_number_of_points_per_voxel,
        max_voxels=20000)
    return voxel_generator
