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
 *
 * Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
 */

#include <glog/logging.h>
#include "vart/runner_ext.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include "xir/tensor/tensor.hpp"
namespace vart {

std::unique_ptr<RunnerExt> RunnerExt::create_runner(
    const xir::Subgraph* subgraph, xir::Attrs* attrs) {
  auto runner = vart::Runner::create_runner_with_attrs(subgraph, attrs);
  auto runner_ext = dynamic_cast<vart::RunnerExt*>(runner.get());
  runner.release();
  CHECK(runner_ext != nullptr) << "cannot create vart::RunnerExt !";
  return std::unique_ptr<vart::RunnerExt>(runner_ext);
}

float to_scale(const xir::Tensor* tensor, float x) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return std::exp2f(x * 1.0f * (float)fixpos);
}

std::vector<float> to_scale(std::vector<const xir::Tensor*> tensors, float x) {
  auto ret = std::vector<float>{};
  ret.reserve(tensors.size());
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(ret),
                 [x](auto& tensor) {
                   int fixpos = tensor->template get_attr<int>("fix_point");
                   auto ret = std::exp2f(x * 1.0f * (float)fixpos);
                   return ret;
                 });
  return ret;
}
float get_input_scale(const xir::Tensor* tensor) {
  return to_scale(tensor, 1.0f);
}
float get_output_scale(const xir::Tensor* tensor) {
  return to_scale(tensor, -1.0f);
}

std::vector<float> get_input_scale(
    std::vector<const xir::Tensor*> input_tensors) {
  return to_scale(input_tensors, 1.0f);
}
std::vector<float> get_output_scale(
    std::vector<const xir::Tensor*> output_tensors) {
  return to_scale(output_tensors, -1.0f);
}

}  // namespace vart
