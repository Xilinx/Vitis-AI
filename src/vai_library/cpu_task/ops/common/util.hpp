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
#pragma once
#include <glog/logging.h>

#include <cmath>
#include <string>

namespace vitis {
namespace ai {
namespace cpu_task {
namespace util {

enum class NONLINEAR { NONE, RELU, PRELU, LEAKYRELU, RELU6, HSIGMOID, HSWISH };

enum NONLINEAR get_nonlinear(const std::string& nonlinear) {
  enum NONLINEAR ret = NONLINEAR::NONE;
  if (nonlinear == "NONE" || nonlinear == "") {
    ret = NONLINEAR::NONE;
  } else if (nonlinear == "RELU") {
    ret = NONLINEAR::RELU;
  } else if (nonlinear == "PRELU") {
    ret = NONLINEAR::PRELU;
  } else if (nonlinear == "LEAKYRELU") {
    ret = NONLINEAR::LEAKYRELU;
  } else if (nonlinear == "RELU6") {
    ret = NONLINEAR::RELU6;
  } else if (nonlinear == "HSIGMOID") {
    ret = NONLINEAR::HSIGMOID;
  } else if (nonlinear == "HSWISH") {
    ret = NONLINEAR::HSWISH;
  } else {
    LOG(FATAL) << "not supported: " << nonlinear;
  }

  return ret;
}

int fix(float data) {
  auto data_max = 127.0;
  auto data_min = -128.0;

  if (data > data_max) {
    data = data_max;
  } else if (data < data_min) {
    data = data_min;
  } else if (data < 0 && (data - floor(data)) == 0.5) {
    data = static_cast<float>(ceil(data));
  } else {
    data = static_cast<float>(round(data));
  }

  return (int)data;
}

std::vector<int> get_dim_stride_vec(const std::vector<int>& in_shape) {
  auto dims = (int)in_shape.size();
  std::vector<int> size_vec(dims);
  auto size = 1;
  size_vec[dims - 1] = 1;
  for (auto idx = dims - 1; idx > 0; --idx) {
    size *= in_shape[idx];
    size_vec[idx - 1] = size;
  }

  return size_vec;
}

}  // namespace util
}  // namespace cpu_task
}  // namespace ai
}  // namespace vitis
