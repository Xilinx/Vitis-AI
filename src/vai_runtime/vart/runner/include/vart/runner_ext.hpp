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
   
   Sample code:

   @code
    auto runner = vart::RunnerExt::create_runner(subgraph, attrs);
    auto input_tensor_buffers = runner->get_inputs();
        for (auto input : input_tensor_buffers) {
            auto shape = input->get_tensor()->get_shape();
    }
   @endcode
   */
  virtual std::vector<vart::TensorBuffer*> get_inputs() = 0;
  /**
   *@brief Gets all output TensorBuffers of RunnerExt.
   *@return All output TensorBuffers. A vector of raw pointer to the output TensorBuffer.
   
   Sample code:

   @code
    auto runner = vart::RunnerExt::create_runner(subgraph, attrs);
    auto output_tensor_buffers = runner->get_outputs();
        for (auto output : output_tensor_buffers) {
            auto shape = output->get_tensor()->get_shape();
    }
   @endcode
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
