/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "vai/tensor_buffer.hpp"
#include "vai/runner.hpp"

#include <cstring>
#include <vector>

namespace vitis {
namespace ai {

class DpuRunner : public Runner<const std::vector<TensorBuffer *> &> {
public:
  enum class TensorFormat { NCHW = 0, NHWC };

public:
  virtual ~DpuRunner() = default;

  virtual std::pair<uint32_t, int>
  execute_async(const std::vector<TensorBuffer *> &input,
                const std::vector<TensorBuffer *> &output) = 0;

  virtual int wait(int jobid, int timeout) = 0;

  virtual TensorFormat get_tensor_format() = 0;

  virtual std::vector<Tensor *> get_input_tensors() = 0;

  virtual std::vector<Tensor *> get_output_tensors() = 0;

  static std::vector<std::unique_ptr<vitis::ai::DpuRunner>> create_dpu_runner(const std::string& model_directory);
};
}
}

extern "C" {
  struct DpuPyTensor
  {
    DpuPyTensor(const vitis::ai::Tensor &tensor) {
      name = new char[tensor.get_name().size()+1];
      std::strcpy(name, tensor.get_name().c_str());
      ndims = tensor.get_dim_num();
      dims = new int[ndims];
      for (int i=0; i < ndims; i++)
        dims[i] = tensor.get_dim_size(i);
      dtype = int(tensor.get_data_type());
    }
    DpuPyTensor(const DpuPyTensor &src) {
      this->name = (char*)malloc(strlen(src.name)+1);
      memset(this->name, 0x0, strlen(src.name)+1);
      strcpy(this->name, src.name);
      this->ndims = src.ndims;
      this->dtype = src.dtype;
      this->dims = new int[ndims];
      for (int i=0; i < ndims; i++)
        dims[i] = src.dims[i];
    }
    ~DpuPyTensor() {
      delete [] name;
      delete [] dims;
    }
    char *name;
    int *dims;
    int ndims;
    int dtype;
  };
} // extern "C"
