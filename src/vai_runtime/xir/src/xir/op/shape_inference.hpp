/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <iostream>
#include <tuple>
#include <vector>

namespace xir {

void shape_infer_data(xir::Op* cur);
void shape_infer_const(xir::Op* cur);
void shape_infer_unsupported(xir::Op* cur);
void shape_infer_conv1d(xir::Op* cur);
void shape_infer_conv2d(xir::Op* cur);
void shape_infer_depthwise_conv2d(xir::Op* cur);
void shape_infer_transposed_conv2d(xir::Op* cur);
void shape_infer_transposed_depthwise_conv2d(xir::Op* cur);
void shape_infer_conv3d(xir::Op* cur);
void shape_infer_depthwise_conv3d(xir::Op* cur);
void shape_infer_transposed_conv3d(xir::Op* cur);
void shape_infer_transposed_depthwise_conv3d(xir::Op* cur);
void shape_infer_pool1d(xir::Op* cur);
void shape_infer_maxpool1d(xir::Op* cur);
void shape_infer_pool2d(xir::Op* cur);
void shape_infer_maxpool2d(xir::Op* cur);
void shape_infer_avgpool2d(xir::Op* cur);
void shape_infer_add(xir::Op* cur);
void shape_infer_sub(xir::Op* cur);
void shape_infer_mul(xir::Op* cur);
void shape_infer_div(xir::Op* cur);
void shape_infer_min(xir::Op* cur);
void shape_infer_max(xir::Op* cur);
void shape_infer_argmax(xir::Op* cur);
void shape_infer_argmax_fix(xir::Op* cur);
void shape_infer_relu(xir::Op* cur);
void shape_infer_leaky_relu(xir::Op* cur);
void shape_infer_prelu(xir::Op* cur);
void shape_infer_relu6(xir::Op* cur);
void shape_infer_elu(xir::Op* cur);
void shape_infer_celu(xir::Op* cur);
void shape_infer_selu(xir::Op* cur);
void shape_infer_gelu(xir::Op* cur);
void shape_infer_mish(xir::Op* cur);
void shape_infer_sigmoid(xir::Op* cur);
void shape_infer_swish(xir::Op* cur);
void shape_infer_hard_sigmoid(xir::Op* cur);
void shape_infer_hard_swish(xir::Op* cur);
void shape_infer_hard_tanh(xir::Op* cur);
void shape_infer_tanh(xir::Op* cur);
void shape_infer_fix(xir::Op* cur);
void shape_infer_float2fix(xir::Op* cur);
void shape_infer_fix2float(xir::Op* cur);
void shape_infer_threshold(xir::Op* cur);
void shape_infer_cast(xir::Op* cur);
void shape_infer_reduction_mean(xir::Op* cur);
void shape_infer_reduction_product(xir::Op* cur);
void shape_infer_reduction_sum(xir::Op* cur);
void shape_infer_reduction_max(xir::Op* cur);
void shape_infer_reduction_max_fix(xir::Op* cur);
void shape_infer_l2_normalize(xir::Op* cur);
void shape_infer_identity(xir::Op* cur);
void shape_infer_placeholder(xir::Op* cur);
void shape_infer_upload(xir::Op* cur);
void shape_infer_download(xir::Op* cur);
void shape_infer_shape(xir::Op* cur);
void shape_infer_reshape(xir::Op* cur);
void shape_infer_squeeze(xir::Op* cur);
void shape_infer_transpose(xir::Op* cur);
void shape_infer_flatten(xir::Op* cur);
void shape_infer_resize(xir::Op* cur);
void shape_infer_inner_product(xir::Op* cur);
void shape_infer_concat(xir::Op* cur);
void shape_infer_reorg(xir::Op* cur);
void shape_infer_softmax(xir::Op* cur);
void shape_infer_cast(xir::Op* cur);
void shape_infer_pad(xir::Op* cur);
void shape_infer_batchnorm(xir::Op* cur);
void shape_infer_instancenorm(xir::Op* cur);
void shape_infer_groupnorm(xir::Op* cur);
void shape_infer_strided_slice(xir::Op* cur);
void shape_infer_priorbox(xir::Op* cur);
void shape_infer_stack(xir::Op* cur);
void shape_infer_matmul(xir::Op* cur);
void shape_infer_gstiling(xir::Op* cur);
void shape_infer_pixel_shuffle(xir::Op* cur);
void shape_infer_exp(xir::Op* cur);
void shape_infer_neg(xir::Op* cur);
void shape_infer_scale(xir::Op* cur);
void shape_infer_correlation2d_elemwise(xir::Op* cur);
void shape_infer_correlation1d_elemwise(xir::Op* cur);
void shape_infer_cost_volume(xir::Op* cur);

// historical
void shape_infer_eltwise(xir::Op* cur);
void shape_infer_final(xir::Op* cur);
void shape_infer_space_to_batch_nd(xir::Op* cur);
void shape_infer_batch_to_space_nd(xir::Op* cur);
void shape_infer_ddr_flatten_concat(xir::Op* cur);

// shape_infer function for fixed ops
void shape_infer_conv2d_fix(xir::Op* cur);
void shape_infer_depthwise_conv2d_fix(xir::Op* cur);
void shape_infer_depthwise_conv3d_fix(xir::Op* cur);
void shape_infer_transposed_conv2d_fix(xir::Op* cur);
void shape_infer_transposed_depthwise_conv2d_fix(xir::Op* cur);
void shape_infer_conv3d_fix(xir::Op* cur);
void shape_infer_transposed_conv3d_fix(xir::Op* cur);
void shape_infer_transposed_depthwise_conv3d_fix(xir::Op* cur);
void shape_infer_const_fix(xir::Op* cur);
void shape_infer_data_fix(xir::Op* cur);
void shape_infer_split_fix(xir::Op* cur);
void shape_infer_eltwise_fix(xir::Op* cur);
void shape_infer_depthwise_fix(xir::Op* cur);
void shape_infer_pool_fix(xir::Op* cur);
void shape_infer_concat_fix(xir::Op* cur);
void shape_infer_reorg_fix(xir::Op* cur);
void shape_infer_ddr_flatten_concat_fix(xir::Op* cur);
void shape_infer_reshape_fix(xir::Op* cur);
void shape_infer_tile_fix(xir::Op* cur);
void shape_infer_pixel_shuffle_fix(xir::Op* cur);
void shape_infer_pad_fix(xir::Op* cur);
void shape_infer_upsample_fix(xir::Op* cur);
void shape_infer_downsample_fix(xir::Op* cur);

// helper function
std::tuple<bool, std::vector<std::int32_t>> size_broadcast(
    const std::vector<std::int32_t>& in_a,
    const std::vector<std::int32_t>& in_b);
std::vector<std::int32_t> flatten_dims(const std::vector<std::int32_t>& dims,
                                       const std::int32_t& start,
                                       const std::int32_t& end);

}  // namespace xir
