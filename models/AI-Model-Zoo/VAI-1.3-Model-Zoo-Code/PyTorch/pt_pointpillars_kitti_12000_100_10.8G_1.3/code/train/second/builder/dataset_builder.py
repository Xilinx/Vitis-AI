
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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""

from second.protos import input_reader_pb2
from second.data.dataset import KittiDataset
from second.data.preprocess import prep_pointcloud
import numpy as np
from second.builder import dbsampler_builder
from functools import partial


def build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          target_assigner=None):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    generate_bev = model_config.use_bev
    without_reflectivity = model_config.without_reflectivity
    num_point_features = model_config.num_point_features
    out_size_factor = model_config.rpn.layer_strides[0] // model_config.rpn.upsample_strides[0]

    cfg = input_reader_config
    db_sampler_cfg = input_reader_config.database_sampler
    db_sampler = None
    if len(db_sampler_cfg.sample_groups) > 0:  # enable sample
        db_sampler = dbsampler_builder.build(db_sampler_cfg)
    u_db_sampler_cfg = input_reader_config.unlabeled_database_sampler
    u_db_sampler = None
    if len(u_db_sampler_cfg.sample_groups) > 0:  # enable sample
        u_db_sampler = dbsampler_builder.build(u_db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    # [352, 400]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    prep_func = partial(
        prep_pointcloud,
        root_path=cfg.kitti_root_path,
        class_names=list(cfg.class_names),
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels=cfg.max_number_of_voxels,
        remove_outside_points=False,
        remove_unknown=cfg.remove_unknown_examples,
        create_targets=training,
        shuffle_points=cfg.shuffle_points,
        gt_rotation_noise=list(cfg.groundtruth_rotation_uniform_noise),
        gt_loc_noise_std=list(cfg.groundtruth_localization_noise_std),
        global_rotation_noise=list(cfg.global_rotation_uniform_noise),
        global_scaling_noise=list(cfg.global_scaling_uniform_noise),
        global_loc_noise_std=(0.2, 0.2, 0.2),
        global_random_rot_range=list(
            cfg.global_random_rotation_range_per_object),
        db_sampler=db_sampler,
        unlabeled_db_sampler=u_db_sampler,
        generate_bev=generate_bev,
        without_reflectivity=without_reflectivity,
        num_point_features=num_point_features,
        anchor_area_threshold=cfg.anchor_area_threshold,
        gt_points_drop=cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=cfg.remove_points_after_sample,
        remove_environment=cfg.remove_environment,
        use_group_id=cfg.use_group_id,
        out_size_factor=out_size_factor)
    dataset = KittiDataset(
        info_path=cfg.kitti_info_path,
        root_path=cfg.kitti_root_path,
        num_point_features=num_point_features,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func)

    return dataset
