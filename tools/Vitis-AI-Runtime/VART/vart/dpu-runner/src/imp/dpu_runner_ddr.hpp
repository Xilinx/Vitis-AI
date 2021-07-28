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
#include <memory>

#include "../dpu_runner_base_imp.hpp"
#include "./dpu_kernel_ddr.hpp"

namespace vart {
namespace dpu {

class DpuRunnerDdr : public DpuRunnerBaseImp {
 public:
  explicit DpuRunnerDdr(const std::vector<const xir::Tensor*> input_tensors,
                        const std::vector<const xir::Tensor*> output_tensors,
                        DpuSessionBaseImp* session);
  DpuRunnerDdr(const DpuRunnerDdr&) = delete;
  DpuRunnerDdr& operator=(const DpuRunnerDdr& other) = delete;
  virtual ~DpuRunnerDdr() = default;

 private:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;

 private:
  virtual void fill_gen_reg(size_t device_core_id,
                            std::vector<uint64_t>& gen_reg) override;

 private:
  std::vector<vart::TensorBuffer*> prepare_input(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output);
  void maybe_copy_input(vart::TensorBuffer::location_t location,
                        const std::vector<vart::TensorBuffer*>& input);
  void prepare_input_for_reg(
      vart::TensorBuffer::location_t location,
      const std::vector<vart::TensorBuffer*>& tensor_buffers,
      std::vector<vart::TensorBuffer*>& ret);

  void prepare_output(const std::vector<vart::TensorBuffer*>& output);
  void copy_data_for_input(vart::TensorBuffer* tb_from,
                           vart::TensorBuffer* tb_to);
  void copy_data_for_output(vart::TensorBuffer* tb_to,
                            vart::TensorBuffer* tb_from);

 private:
  std::vector<vart::TensorBuffer*> my_input_;
};

}  // namespace dpu
}  // namespace vart
