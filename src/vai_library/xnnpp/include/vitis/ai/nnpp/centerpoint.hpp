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
/*
 * Filename: centerpoint.hpp
 *
 * Description:
 * This network is used to detecting objects from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 *@struct CenterPointResult
 *@brief Struct of the result with the centerpoint network.
 *
 */
struct CenterPointResult {
  /// Bounding box 3d: {x, y, z, x_size, y_size, z_size, yaw}
  std::vector<float> bbox;
  /// Score
  float score;
  /// Classification
  int label;
};

std::vector<CenterPointResult> post_process(
    std::vector<std::vector<vitis::ai::library::OutputTensor>> out_tensor,
    size_t batch_ind);

}  // namespace ai
}  // namespace vitis
