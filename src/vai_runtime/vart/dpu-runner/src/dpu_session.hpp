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
#include <algorithm>
#include <functional>
#include <memory>
#include <vart/runner.hpp>
#include <vitis/ai/with_injection.hpp>
namespace vart {
namespace dpu {

class DpuSession : public vitis::ai::WithInjection<DpuSession> {
 public:
  explicit DpuSession() = default;
  static std::unique_ptr<DpuSession> create(const std::string& filename,
                                            const std::string& kernel);
  static std::unique_ptr<DpuSession> create(const xir::Subgraph* subgraph,
                                            xir::Attrs* attrs);

 public:
  DpuSession(const DpuSession&) = delete;
  DpuSession& operator=(const DpuSession& other) = delete;

  virtual ~DpuSession() = default;

 public:
  virtual std::unique_ptr<vart::Runner> create_runner() = 0;
  virtual std::vector<vart::TensorBuffer*> get_inputs() = 0;
  virtual std::vector<vart::TensorBuffer*> get_outputs() = 0;
  virtual std::vector<const xir::Tensor*> get_input_tensors() const = 0;
  virtual std::vector<const xir::Tensor*> get_output_tensors() const = 0;
};
}  // namespace dpu
}  // namespace vart
