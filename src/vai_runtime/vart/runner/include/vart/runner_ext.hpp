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
#include <vart/runner.hpp>
namespace vart {

class RunnerExt : public vart::Runner {
 public:
 public:
  explicit RunnerExt() = default;

  RunnerExt(const RunnerExt&) = delete;
  RunnerExt& operator=(const RunnerExt& other) = delete;

  virtual ~RunnerExt() = default;

  /**
   * @brief Factory fucntion to create an instance of runner by
   * subgraph and attrs
   * @param subgraph  XIR Subgraph
   * @param attrs XIR attrs object, this object is shared among all
   * runners on the same graph.
   * @return An instance of runner.
   */
  static std::unique_ptr<RunnerExt> create_runner(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs);

 public:
  /**
   *@brief Gets all input TensorBuffers of RunnerExt.
   *@return All input TensorBuffers. A vector of raw pointer to the input TensorBuffer.
   */
  virtual std::vector<vart::TensorBuffer*> get_inputs() = 0;
  /**
   *@brief Gets all output TensorBuffers of RunnerExt.
   *@return All output TensorBuffers. A vector of raw pointer to the output TensorBuffer.
   */
  virtual std::vector<vart::TensorBuffer*> get_outputs() = 0;
};

std::vector<float> get_input_scale(
    std::vector<const xir::Tensor*> input_tensors);
std::vector<float> get_output_scale(
    std::vector<const xir::Tensor*> output_tensors);
float get_input_scale(const xir::Tensor* input_tensor);
float get_output_scale(const xir::Tensor* output_tensor);

}  // namespace vart
