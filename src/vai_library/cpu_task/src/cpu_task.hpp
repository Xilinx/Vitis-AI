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

#include <unordered_map>

#include "./tensor_buffer_proxy.hpp"
#include "vart/op_imp.h"
#include "vart/runner.hpp"

namespace {
struct MyOpArgs {
  std::vector<vart::OpImpArg> inputs;
  vart::TensorBuffer* output;
};

class CpuTask : public vart::Runner {
 public:
  explicit CpuTask(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  CpuTask(const CpuTask& other) = delete;

  virtual ~CpuTask();

 public:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;

 private:
  std::unordered_map<std::string, size_t> build_tensor_name_2_index();
  std::vector<MyOpArgs> build_my_op_args();

  vart::TensorBuffer* find_tensor_buffer(const std::string& name);
  vart::TensorBufferProxy* find_proxy_tensor_buffer(const std::string& name);
  void update_proxy(const std::vector<vart::TensorBuffer*>& input);
  void maybe_sync_for_read(const std::vector<vart::TensorBuffer*>& b);
  void maybe_sync_for_write(const std::vector<vart::TensorBuffer*>& b);
  void maybe_sync_for_read(vart::TensorBuffer* b);
  void maybe_sync_for_write(vart::TensorBuffer* b);

 private:
  std::unique_ptr<xir::Attrs> default_attrs_;
  xir::Attrs* attrs_;
  const xir::Subgraph* subgraph_;
  std::vector<std::unique_ptr<xir::Tensor>> inputs_;
  std::vector<std::unique_ptr<xir::Tensor>> outputs_;
  std::vector<const xir::Op*> ops_;
  std::vector<std::unique_ptr<vart::OpImp>> op_imp_;
  // all tensors inside the subgraph, not including the input ops
  std::vector<const xir::Tensor*> tensors_;
  // the corresponding tensor buffers, including the input ops.
  std::vector<std::unique_ptr<vart::TensorBuffer>> tensor_buffers_;
  std::vector<std::unique_ptr<vart::TensorBufferProxy>> proxy_tensor_buffers_;
  // pointer to point to OpArgs;
  std::vector<vart::TensorBuffer*> tensor_buffer_view_;
  std::unordered_map<std::string, size_t> tensor_name_2_index_;
  std::vector<MyOpArgs> my_op_args_;
};
}  // namespace
