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

#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <memory>
#include <string>
#include <vector>
#include <vitis/ai/library/tensor.hpp>
using namespace std;

namespace vitis {
namespace ai {

/**
 *@struct X_Autonomous3DResult
 *@brief Struct of the result with the X_Autonomous3D network.
 *
 */
struct X_Autonomous3DResult {
  struct BBox {
    /// Bounding box 3d: {x, y, z, x_size, y_size, z_size, yaw}
    std::vector<float> bbox;
    /// Score
    float score;
    /// Classification
    int label;
  };
  /// All objects, a vector of BBox
  std::vector<BBox> bboxes;
};

/**
 * @class X_Autonomous3DPostProcess
 * @brief Class of the X_Autonomous3D post-process. It initializes the
 * parameters once instead of computing them each time the program executes.
 * */

class X_Autonomous3DPostProcess {
 public:
  /**
   * @brief Create an X_Autonomous3DPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The DPU model configuration information.
   * @return An unique pointer of X_Autonomous3DPostProcess.
   */

  static std::unique_ptr<X_Autonomous3DPostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);

  /**
   * @brief The batch mode post-processing function of the X_Autonomous3D
   * network.
   * @return The vector of struct of X_Autonomous3DResult.
   */

  virtual std::vector<X_Autonomous3DResult> process(size_t batch_size) = 0;

  /**
   * @cond NOCOMMENTS
   */
  virtual ~X_Autonomous3DPostProcess();

 protected:
  explicit X_Autonomous3DPostProcess();
  X_Autonomous3DPostProcess(const X_Autonomous3DPostProcess&) = delete;
  X_Autonomous3DPostProcess& operator=(const X_Autonomous3DPostProcess&) =
      delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis

