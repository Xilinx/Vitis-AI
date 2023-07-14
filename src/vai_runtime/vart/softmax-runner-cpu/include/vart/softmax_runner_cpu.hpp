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

#include <memory>

#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"
#include "xir/sfm_controller.hpp"

namespace vart {
class SoftmaxRunnerCPU : public vart::RunnerExt {
 public:
  explicit SoftmaxRunnerCPU(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  SoftmaxRunnerCPU(const SoftmaxRunnerCPU& other) = delete;

  virtual ~SoftmaxRunnerCPU();

 public:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;
  virtual std::vector<vart::TensorBuffer*> get_inputs() override;
  virtual std::vector<vart::TensorBuffer*> get_outputs() override;

 private:
  std::unique_ptr<vart::TensorBuffer> input_;
  std::unique_ptr<vart::TensorBuffer> output_;
};
}  // namespace vart
