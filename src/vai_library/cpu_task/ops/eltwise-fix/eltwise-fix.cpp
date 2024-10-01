/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include <cmath>
#include <iostream>

#include "../common/util.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

using namespace std;

namespace {
enum EltwiseType { ADD, MUL };

static enum EltwiseType get_eltwise_type(const std::string& type) {
  auto ret = ADD;
  if (type == "ADD") {
  } else if (type == "MUL") {
    ret = MUL;
  } else {
    LOG(FATAL) << "unsupported eltwise type: " << type;
    exit(1);
  }
  return ret;
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    if (op->has_attr("type")) {
      type_ = get_eltwise_type(op->get_attr<string>("type"));
    }
    if (op->has_attr("nonlinear")) {
      nonlinear_ = vitis::ai::cpu_task::util::get_nonlinear(
          op->get_attr<string>("nonlinear"));
    }
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                std::vector<vart::simple_tensor_buffer_t<int8_t>> inputs) {
    // output info
    auto input_num = static_cast<int>(inputs.size());
    output_shape_ = output.tensor->get_shape();
    auto output_dim_size = output_shape_.size();
    output_shape_stride_vec_ =
        vitis::ai::cpu_task::util::get_dim_stride_vec(output_shape_);
    auto element_num = output.tensor->get_element_num();

    // input info
    std::vector<int> input_fp_vec(input_num, 0);
    std::vector<int8_t*> input_ptr_vec(input_num, nullptr);
    int input_max_fp = 0;
    for (auto idx = 0; idx < input_num; ++idx) {
      // get fix_point for each input and find the max one
      auto in_fp = inputs[idx].tensor->get_attr<int>("fix_point");
      input_fp_vec[idx] = in_fp;
      if (input_max_fp < in_fp) input_max_fp = in_fp;

      // collect input data pointer
      input_ptr_vec[idx] = inputs[idx].data;

      // for the inputs with different dimensions, need to broadcast
      // here extend shape first
      auto in_shape = inputs[idx].tensor->get_shape();
      std::vector<int> new_in_shape(output_dim_size, 1);
      if (in_shape.size() == output_dim_size) {
        new_in_shape = in_shape;
      } else {
        std::copy(in_shape.rbegin(), in_shape.rend(), new_in_shape.rbegin());
      }
      // most cases should be the same shape,
      // for better performance, the inputs are divided into two containers
      if (check_same_shape(new_in_shape)) {
        input_same_shape_vec_.emplace_back(idx);
      } else {
        input_diff_shape_map_.emplace(idx, new_in_shape);
      }
    }

    // input scale
    input_scale_vec_.reserve(input_num);
    for (auto& ifp : input_fp_vec) {
      input_scale_vec_.emplace_back(pow(2.0f, (input_max_fp - ifp)));
    }

    // output scale
    auto output_fp = output.tensor->get_attr<int>("fix_point");
    output_scale_ = pow(2.0f, output_fp - input_max_fp);
    six_as_output_ = 6.0f * std::pow(2.0f, output_fp);

    switch (type_) {
      case ADD:
        eltwise_add(input_ptr_vec, output.data, element_num);
        break;
      case MUL:
        eltwise_mul(input_ptr_vec, output.data, element_num);
        break;
      default:
        exit(1);
    }
    return 0;
  }

 private:
  void eltwise_add(const std::vector<int8_t*>& input, int8_t* output,
                   int element_num) {
    auto empty_map = input_diff_shape_map_.empty();

    for (auto idx = 0; idx < element_num; ++idx) {
      float data = 0;
      // process inputs with same shape as output
      // most cases are covered in this loop
      for (auto& index : input_same_shape_vec_) {
        // add every element
        data += input[index][idx] * input_scale_vec_[index];
      }
      // process inputs with different shape from output
      // get the mapped input offset by output element index
      if (!empty_map) {
        for (auto it = input_diff_shape_map_.begin();
             it != input_diff_shape_map_.end(); ++it) {
          auto key = it->first;
          auto in_shape = it->second;
          auto in_offset = get_input_ptr_offset(in_shape, idx);
          data += input[key][in_offset] * input_scale_vec_[key];
        }
      }
      output[idx] =
          vitis::ai::cpu_task::util::fix(nonlinear(data * output_scale_));
    }
    return;
  }
  void eltwise_mul(const std::vector<int8_t*>& input, int8_t* output,
                   int element_num) {
    auto empty_map = input_diff_shape_map_.empty();

    for (auto idx = 0; idx < element_num; ++idx) {
      float data = 1;
      // process inputs with same shape as output
      // most cases are covered in this loop
      for (auto& index : input_same_shape_vec_) {
        // add every element
        data *= input[index][idx] * input_scale_vec_[index];
      }
      // process inputs with different shape from output
      // get the mapped input offset by output element index
      if (!empty_map) {
        for (auto it = input_diff_shape_map_.begin();
             it != input_diff_shape_map_.end(); ++it) {
          auto key = it->first;
          auto in_shape = it->second;
          auto in_offset = get_input_ptr_offset(in_shape, idx);
          data *= input[key][in_offset] * input_scale_vec_[key];
        }
      }
      output[idx] =
          vitis::ai::cpu_task::util::fix(nonlinear(data * output_scale_));
    }
    return;
  }

  // activation
  float nonlinear(float in) {
    auto ret = in;
    switch (nonlinear_) {
      case vitis::ai::cpu_task::util::NONLINEAR::RELU:
        if (ret < 0) ret = 0;
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::PRELU:
        if (ret < 0) ret *= 0.0;
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::LEAKYRELU:
        if (ret < 0) ret *= 0.01;
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::RELU6:
        ret = max(0.0f, min(six_as_output_, 6.0f));
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::NONE:
      default:
        break;
    }
    return ret;
  }

  // check if the input shape is same as output shape
  bool check_same_shape(const std::vector<int>& input_shape) {
    // different dimension
    if (input_shape.size() != output_shape_.size()) return false;
    // different shape
    for (auto i = 0; i < (int)output_shape_.size(); ++i) {
      if (input_shape[i] != output_shape_[i]) return false;
    }
    return true;
  }

  // 1. for each dim, parse the dim index, like n,h,w,c
  // 2. % for each dim, n,h,w,c%input_shape_dim
  // 3. input_dim_index*dim_stride to get offset
  int get_input_ptr_offset(const std::vector<int>& in_shape,
                           int output_ptr_offset) {
    int ret = 0;
    int in_shape_size = (int)in_shape.size();
    std::vector<int> index_vec(in_shape.size());
    // 1
    for (auto idx = 0; idx < in_shape_size; ++idx) {
      // 2
      index_vec[idx] =
          (output_ptr_offset / output_shape_stride_vec_[idx]) % in_shape[idx];
      output_ptr_offset %= output_shape_stride_vec_[idx];
    }
    // 3
    auto in_shape_dim_stride_vec =
        vitis::ai::cpu_task::util::get_dim_stride_vec(in_shape);
    for (auto i = 0; i < in_shape_size; ++i) {
      ret += index_vec[i] * in_shape_dim_stride_vec[i];
    }
    return ret;
  }

 public:
  // op attr
  enum EltwiseType type_ { ADD };
  vitis::ai::cpu_task::util::NONLINEAR nonlinear_ = vitis::ai::cpu_task::util::NONLINEAR::NONE;

  // output shape and dim stride size
  std::vector<int> output_shape_;
  std::vector<int> output_shape_stride_vec_;
  // input shape
  std::vector<std::vector<int>> input_shape_vec_;

  // store the index of inputs with same shape as output
  std::vector<int> input_same_shape_vec_;
  // store the index and shape of inputs with different shape from output
  std::map<int, std::vector<int>> input_diff_shape_map_;

  // scales of input and output
  std::vector<float> input_scale_vec_;
  float output_scale_=0.0;
  // for ReLU6
  float six_as_output_=0.0;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
