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
#include "../include/xilinx/ai/nnpp/facequality.hpp"

#include <vector>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/math.hpp>
#include <xilinx/ai/profiling.hpp>

using namespace std;

namespace xilinx {
namespace ai {

FaceQualityResult face_quality_post_process(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  // Valid output size
  int output_tensor_size = 3;
  vector<float> softmaxvalue(output_tensor_size);

  xilinx::ai::softmax((int8_t*)output_tensors[0][0].data,
                      xilinx::ai::tensor_scale(output_tensors[0][0]),
                      output_tensor_size, 1, softmaxvalue.data());
  float quality = 0;
  if (softmaxvalue[1] >= 0.5) {
    quality = softmaxvalue[1];
  } else if (softmaxvalue[2] >= 0.6) {
    quality = (1.0 - softmaxvalue[2]) / 4;
  } else {
    quality =
        0.1 + softmaxvalue[0] / 2.5 + softmaxvalue[1] / 5 - softmaxvalue[2] / 6;
  }

  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;
  return FaceQualityResult{input_width, input_height, quality};
}

}  // namespace ai
}  // namespace xilinx
