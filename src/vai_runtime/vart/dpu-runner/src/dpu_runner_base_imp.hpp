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
#include <vart/runner.hpp>
#include <xir/device_memory.hpp>
#include <xir/graph/graph.hpp>

#include "dpu_kernel.hpp"
#include "dpu_session_base_imp.hpp"
#include "my_tensor.hpp"

namespace vart {
namespace dpu {
class DpuRunnerBaseImp : public vart::Runner {
 public:
 public:
  explicit DpuRunnerBaseImp(
      const std::vector<const xir::Tensor*> input_tensors,
      const std::vector<const xir::Tensor*> output_tensors,
      DpuSessionBaseImp* session);

  DpuRunnerBaseImp(const DpuRunnerBaseImp&) = delete;
  DpuRunnerBaseImp& operator=(const DpuRunnerBaseImp& other) = delete;

  virtual ~DpuRunnerBaseImp();

 private:
  // implementation should fillin the reg setting for workspace
  virtual void fill_gen_reg(size_t device_core_id,
                            std::vector<uint64_t>& gen_reg) = 0;

 private:
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;

  virtual std::vector<const xir::Tensor*> get_output_tensors() override;

 protected:
  void start_dpu2(size_t device_core_id);

 protected:
  const std::vector<const xir::Tensor*> input_tensors_;
  const std::vector<const xir::Tensor*> output_tensors_;
  static void copy_tensor_buffer(vart::TensorBuffer* tb_from,
                                 vart::TensorBuffer* tb_to, float scale);
  bool check_fingerprint(size_t device_core_id);

 public:
  enum TensorType { INPUT, INTERNAL, OUTPUT };

 private:
  void prepare_envirnment(const DpuKernel::SubgraphCode& sg_and_code,
                          const std::vector<uint64_t>& gen_reg,
                          size_t device_core_id);
  void before_run_dpu();
  void after_run_dpu();
  void clear_environment();
  const my_tensor_t& find_tensor(const std::string& name);
  void dump_tensor(const my_tensor_t& tensor);
  void upload_tensor(const my_tensor_t& tensor);
  void clear_tensor(const my_tensor_t& tensor);
  void compare_tensor(const my_tensor_t& tensor);
  typedef void (DpuRunnerBaseImp::*tensor_fun_t)(const my_tensor_t& tensor);
  void for_each_tensor(const std::vector<const xir::Tensor*> tensors,
                       tensor_fun_t f);

  bool update_tensor_data_by_stride(std::vector<char>& buf,
                                    const xir::Tensor* tensor,
                                    const size_t offset);
  bool download_tensor_data_by_stride(std::vector<char>& buf,
                                      const xir::Tensor* tensor,
                                      const size_t offset);
  std::vector<const xir::Tensor*> get_internal_tensor(
      const xir::Subgraph* subgraph);
  std::vector<const xir::Tensor*> get_input_tensor(
      const xir::Subgraph* subgraph);
  std::vector<const xir::Tensor*> get_output_tensor(
      const xir::Subgraph* subgraph);

 private:
  xir::DeviceMemory* get_device_memory();

 protected:
  DpuSessionBaseImp* session_;

 private:
  // all below variables are used to avoid too many parameters for
  // memory functions. they are set before start_dpu() and used by
  // dump/compare/clear/upload tensors.
  std::shared_ptr<xir::DeviceMemory> device_memory_;
  const xir::Subgraph* subgraph_;
  std::vector<uint64_t> regs_;
  //
  std::string tensor_output_dir_ = "unkown";
};

}  // namespace dpu
}  // namespace vart
