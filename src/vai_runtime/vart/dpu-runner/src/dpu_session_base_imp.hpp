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

#include <memory>
#include <xir/dpu_controller.hpp>

#include "./dpu_kernel.hpp"
#include "./dpu_session.hpp"
#include "./my_tensor.hpp"

namespace vart {
namespace dpu {

class DpuSessionBaseImp : public DpuSession {
 public:
  // static int session_count;

 public:
  explicit DpuSessionBaseImp(xir::Attrs* attrs);

  DpuSessionBaseImp(const DpuSessionBaseImp&) = delete;
  DpuSessionBaseImp& operator=(const DpuSessionBaseImp& other) = delete;

  virtual ~DpuSessionBaseImp() = default;

 public:
  // now edge and cloud have the same implementation, because a runner
  // is bound to a device core.
  size_t get_num_of_engines() const;

 protected:
  virtual std::vector<const xir::Tensor*> get_input_tensors() const override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() const override;
  virtual void initialize() override;

 private:
  size_t my_get_device_core_id(size_t cu_size, xir::Attrs* attrs);
  std::vector<my_tensor_t> init_input_tensors(const xir::Subgraph* subgraph);
  std::vector<my_tensor_t> init_output_tensors(const xir::Subgraph* subgraph);
  std::vector<my_tensor_t> init_tensors(
      const xir::Subgraph* subgraph, const std::vector<std::string>& op_names,
      bool check_stride);

 public:
  xir::DpuController* get_dpu_controller() { return dpu_controller_.get(); }
  size_t get_device_core_id() const { return device_core_id_; }
  vart::dpu::DpuKernel* get_kernel() { return kernel_.get(); }
  const std::vector<my_tensor_t>& get_my_input_tensors() const {
    return my_input_tensors_;
  }
  const std::vector<my_tensor_t>& get_my_output_tensors() const {
    return my_output_tensors_;
  }

 protected:
  std::unique_ptr<xir::Attrs> default_attrs_;
  xir::Attrs* attrs_;
  std::vector<my_tensor_t> my_input_tensors_;
  std::vector<my_tensor_t> my_output_tensors_;
  std::vector<my_tensor_t> my_all_tensors_;
  std::shared_ptr<vart::dpu::DpuKernel> kernel_;
  std::shared_ptr<xir::DpuController> dpu_controller_;
  size_t device_core_id_;
  friend class CloudDpuRunner;
  friend class EdgeDpuRunner;
  friend class DpuRunnerBaseImp;
};

}  // namespace dpu
}  // namespace vart
