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
#include <memory>
#include <string>
#include <vart/runner.hpp>
namespace vart {
namespace dpu {
class DpuRunnerExt : public vart::Runner {
 public:
 public:
  explicit DpuRunnerExt() = default;

  DpuRunnerExt(const DpuRunnerExt&) = delete;
  DpuRunnerExt& operator=(const DpuRunnerExt& other) = delete;

  virtual ~DpuRunnerExt() = default;

 public:
  /** @brief return the allocated input tensor buffers.
   *
   * potentially more efficient
   * */
  virtual std::vector<vart::TensorBuffer*> get_inputs() = 0;
  /** @brief return the allocated output tensor buffers.
   *
   * potentially more efficient
   * */
  virtual std::vector<vart::TensorBuffer*> get_outputs() = 0;

  /** @brief return the input scale for conversion from float to fix point
   * */
  virtual std::vector<float> get_input_scale() const = 0;

  /** @brief return the input scale for conversion from fix point to float
   * */
  virtual std::vector<float> get_output_scale() const = 0;
};
}  // namespace dpu
}  // namespace vart
