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

#include <vector>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {

/**
 *@struct RetinaFaceResult
 *@brief Struct of the result with the retinaface network.
 *
 */
struct RetinaFaceResult {
  /**
   *@struct BoundingBox
   *@brief The coordinate and confidence of a face.
   */
  struct BoundingBox {
    /// x-coordinate. x is normalized relative to the input image columns.
    /// Range from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// Range from 0 to 1.
    float y;
    /// Face width. Width is normalized relative to the input image columns,
    /// Range from 0 to 1.
    float width;
    /// Face height. Heigth is normalized relative to the input image rows,
    /// Range from 0 to 1.
    float height;
    /// Face confidence. The value ranges from 0 to 1.
    float score;
  };
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// All faces, filtered by confidence >= detect threshold.
  std::vector<BoundingBox> bboxes;
  /// Landmarks
  std::vector<std::array<std::pair<float, float>, 5>> landmarks;
};

/**
 * @class RetinaFacePostProcess
 * @brief Class of the retinaface post-process. It initializes the parameters
 * once instead of computing them each time the program executes.
 * */
class RetinaFacePostProcess {
 public:
  /**
   * @brief Create a RetinaFacePostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The DPU model configuration information.
   * @return A unique pointer of RetinaFacePostProcess.
   */
  static std::unique_ptr<RetinaFacePostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);

  /**
   * @brief The batch mode post-processing function of the retinaface network.
   * @return The vector of struct of RetinaFaceResult.
   */
  virtual std::vector<RetinaFaceResult> retinaface_post_process(
      size_t batch_size) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual ~RetinaFacePostProcess();

 protected:
  explicit RetinaFacePostProcess();
  RetinaFacePostProcess(const RetinaFacePostProcess&) = delete;
  RetinaFacePostProcess& operator=(const RetinaFacePostProcess&) = delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
