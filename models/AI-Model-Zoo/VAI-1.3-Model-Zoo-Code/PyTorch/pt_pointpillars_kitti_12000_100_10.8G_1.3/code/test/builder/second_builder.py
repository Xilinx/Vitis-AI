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


# Copyright 2017 yanyan. All Rights Reserved.
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
"""VoxelNet builder.
"""

from models.voxelnet import LossNormType, VoxelNet, PostProcess, PreProcess

def build(model_cfg, voxel_generator,
          target_assigner) -> VoxelNet:
    """build second pytorch instance.
    """
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    vfe_with_distance = model_cfg.voxel_feature_extractor.with_distance
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_class = model_cfg.num_class

    num_input_features = model_cfg.num_point_features
    if model_cfg.without_reflectivity:
        num_input_features = 3
    loss_norm_type_dict = {
        0: LossNormType.NormByNumExamples,
        1: LossNormType.NormByNumPositives,
        2: LossNormType.NormByNumPosNeg,
    }
    loss_norm_type = loss_norm_type_dict[model_cfg.loss_norm_type]

    # losses = losses_builder.build(model_cfg.loss)
    losses = None
    encode_rad_error_by_sin = model_cfg.encode_rad_error_by_sin
    # cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight = None, None, None, None
    pos_cls_weight = model_cfg.pos_class_weight
    neg_cls_weight = model_cfg.neg_class_weight
    direction_loss_weight = model_cfg.direction_loss_weight

    net = VoxelNet(
        dense_shape,
        num_class=num_class,
        vfe_class_name=model_cfg.voxel_feature_extractor.module_class_name,
        vfe_num_filters=vfe_num_filters,
        middle_class_name=model_cfg.middle_feature_extractor.module_class_name,
        middle_num_filters_d1=list(
            model_cfg.middle_feature_extractor.num_filters_down1),
        middle_num_filters_d2=list(
            model_cfg.middle_feature_extractor.num_filters_down2),
        rpn_class_name=model_cfg.rpn.module_class_name,
        rpn_layer_nums=list(model_cfg.rpn.layer_nums),
        rpn_layer_strides=list(model_cfg.rpn.layer_strides),
        rpn_num_filters=list(model_cfg.rpn.num_filters),
        rpn_upsample_strides=list(model_cfg.rpn.upsample_strides),
        rpn_num_upsample_filters=list(model_cfg.rpn.num_upsample_filters),
        use_norm=True,
        use_rotate_nms=model_cfg.use_rotate_nms,
        multiclass_nms=model_cfg.use_multi_class_nms,
        nms_score_threshold=model_cfg.nms_score_threshold,
        nms_pre_max_size=model_cfg.nms_pre_max_size,
        nms_post_max_size=model_cfg.nms_post_max_size,
        nms_iou_threshold=model_cfg.nms_iou_threshold,
        use_sigmoid_score=model_cfg.use_sigmoid_score,
        encode_background_as_zeros=model_cfg.encode_background_as_zeros,
        use_direction_classifier=model_cfg.use_direction_classifier,
        use_bev=model_cfg.use_bev,
        num_input_features=num_input_features,
        num_groups=model_cfg.rpn.num_groups,
        use_groupnorm=model_cfg.rpn.use_groupnorm,
        with_distance=vfe_with_distance,
        cls_loss_weight=cls_weight,
        loc_loss_weight=loc_weight,
        pos_cls_weight=pos_cls_weight,
        neg_cls_weight=neg_cls_weight,
        direction_loss_weight=direction_loss_weight,
        loss_norm_type=loss_norm_type,
        encode_rad_error_by_sin=encode_rad_error_by_sin,
        loc_loss_ftor=loc_loss_ftor,
        cls_loss_ftor=cls_loss_ftor,
        target_assigner=target_assigner,
        voxel_size=voxel_generator.voxel_size,
        pc_range=voxel_generator.point_cloud_range
    )

    preprocessor = PreProcess(
        with_distance=vfe_with_distance,
        voxel_size=voxel_generator.voxel_size,
        pc_range=voxel_generator.point_cloud_range)

    postprocessor = PostProcess(
        num_class=num_class,
        use_direction_classifier=model_cfg.use_direction_classifier,
        use_sigmoid_score=model_cfg.use_sigmoid_score,
        encode_background_as_zeros=model_cfg.encode_background_as_zeros,
        use_rotate_nms=model_cfg.use_rotate_nms,
        multiclass_nms=model_cfg.use_multi_class_nms,
        nms_score_threshold=model_cfg.nms_score_threshold,
        nms_pre_max_size=model_cfg.nms_pre_max_size,
        nms_post_max_size=model_cfg.nms_post_max_size,
        nms_iou_threshold=model_cfg.nms_iou_threshold,
        target_assigner=target_assigner
        )
    return net, preprocessor, postprocessor
