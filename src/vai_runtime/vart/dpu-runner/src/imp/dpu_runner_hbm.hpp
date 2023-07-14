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
#include <xir/device_memory.hpp>

#include "../dpu_kernel.hpp"
#include "../dpu_runner_base_imp.hpp"
#include "../my_tensor.hpp"
#include "dpu_core.hpp"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace vart {
namespace dpu {

class DpuRunnerHbm : public DpuRunnerBaseImp {
 public:
  explicit DpuRunnerHbm(const std::vector<const xir::Tensor*> input_tensors,
                        const std::vector<const xir::Tensor*> output_tensors,
                        const size_t core_id, DpuSessionBaseImp* session);
  DpuRunnerHbm(const DpuRunnerHbm&) = delete;
  DpuRunnerHbm& operator=(const DpuRunnerHbm& other) = delete;

  virtual ~DpuRunnerHbm();

 private:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;

 private:
  virtual void fill_gen_reg(size_t device_core_id,
                            std::vector<uint64_t>& gen_reg) override;

 private:
  void upload_data(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>&
          chunks,  // chunks[engine_id][reg_id].get();
      size_t device_core_id);
  std::vector<uint64_t> start_dpu(
      size_t device_core_id,
      const std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>&
          chunks,  // chunks[engine_id][reg_id].get();
      DpuCoreWorkspace* w);
  void download_data(
      const std::vector<vart::TensorBuffer*>& output,
      const std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>&
          chunks,  // chunks[engine_id][reg_id].get();
      size_t device_core_id);

 private:
  std::vector<std::shared_ptr<DpuCore>> cores_;
  const size_t device_core_id_;
  // local temp variable for fillin registers of workspaces
  std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>> chunks_;
  std::shared_ptr<xir::DeviceMemory> device_memory_;
};

}  // namespace dpu
}  // namespace vart
