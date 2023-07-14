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
#include <memory>
#include <xir/dpu_controller.hpp>

#include "../dpu_session_base_imp.hpp"
#include "dpu_kernel.hpp"
namespace vart {
namespace dpu {

class DpuSessionImp : public vart::dpu::DpuSessionBaseImp {
 public:
  explicit DpuSessionImp(const std::string& filename,
                         const std::string& kernel);
  explicit DpuSessionImp(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  DpuSessionImp(const DpuSessionImp&) = delete;
  DpuSessionImp& operator=(const DpuSessionImp& other) = delete;

  virtual ~DpuSessionImp();

 public:
  virtual const std::vector<vart::TensorBuffer*>& get_reg_base() {
    return reg_base_;
  }

 private:
  virtual void initialize() override;
  virtual std::unique_ptr<vart::Runner> create_runner() override;
  virtual std::vector<vart::TensorBuffer*> get_inputs() override;
  virtual std::vector<vart::TensorBuffer*> get_outputs() override;

 private:
  void set_subgraph_specific_attrs();
  std::vector<std::unique_ptr<vart::TensorBuffer>> init_tensor_buffer(
      std::vector<my_tensor_t>& tensors);
  std::vector<vart::TensorBuffer*> find_tensor_buffer(
      const std::vector<std::string>& names);
  std::vector<vart::TensorBuffer*> find_reg_tensor_buffer();

 private:
  std::vector<std::unique_ptr<vart::TensorBuffer>> all_tensor_buffers_;
  std::vector<vart::TensorBuffer*> input_tensor_buffers_;
  std::vector<vart::TensorBuffer*> output_tensor_buffers_;
  std::vector<vart::TensorBuffer*> reg_base_;
};

}  // namespace dpu
}  // namespace vart
