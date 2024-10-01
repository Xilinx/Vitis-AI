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

#include <cstring>
#include <memory>
#include <vector>

#include "./runner.hpp"
#include "./tensor_buffer.hpp"

namespace vitis {
namespace ai {

class DpuRunner : public Runner<const std::vector<TensorBuffer*>&> {
  //
 public:
  /** @brief create dpu runner
   *
   * because one DPU model may contains more than one DPU kernels(or DPU
   * subgraph), this function returns a vector of a dpu runner.
   *
   * @param model_directory the directory name which contrains meta.json
   * */
  static std::vector<std::unique_ptr<vitis::ai::DpuRunner>> create_dpu_runner(
      const std::string& model_directory);

 public:
  enum class TensorFormat { NCHW = 0, NHWC };

 public:
  virtual ~DpuRunner() = default;

  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<TensorBuffer*>& input,
      const std::vector<TensorBuffer*>& output) = 0;

  virtual int wait(int jobid, int timeout) = 0;

  virtual TensorFormat get_tensor_format() = 0;
  // TODO: shall it be  std::vector<const Tensor*> ?
  virtual std::vector<Tensor*> get_input_tensors() = 0;

  virtual std::vector<Tensor*> get_output_tensors() = 0;
};
}  // namespace ai
}  // namespace vitis

extern "C" {
struct DpuPyTensor {
  DpuPyTensor(const vitis::ai::Tensor& tensor) {
    name = tensor.get_name().c_str();
    dims = tensor.get_dims().data();
    ndims = tensor.get_dim_num();
    dtype = std::int32_t(tensor.get_data_type());
  }

  const char* name;
  const std::int32_t* dims;
  std::int32_t ndims;
  std::int32_t dtype;
};
void* DpuPyRunnerCreate(char* path);
void DpuPyRunnerGetInputTensors(void* runner, DpuPyTensor** tensors,
                                int* tensor_cnt);
void DpuPyRunnerGetOutputTensors(void* runner, DpuPyTensor** tensors,
                                 int* tensor_cnt);
int DpuPyRunnerGetTensorFormat(void* runner);
int DpuPyRunnerExecuteAsync(void* runner, void** indata, void** outdata,
                            int batch_sz, int* status);
int DpuPyRunnerWait(void* runner, int jobId);
void DpuPyRunnerDestroy(void* runner);
}  // extern "C"
