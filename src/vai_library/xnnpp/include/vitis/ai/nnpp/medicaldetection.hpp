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
 * Filename: medicaldetection.hpp
 *
 * Description:
 * This network is used to detecting objects from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 * @struct MedicalDetectionResult
 * @brief Struct of the result returned by the medical refinedet network.
 */
struct MedicalDetectionResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /**
   * @struct BoundingBox
   * @brief  Struct of an object coordinate ,confidence and classification.
   */
  struct BoundingBox {
    /// Classification
    int label;
    /// Confidence
    float score;
    /// x-coordinate. x is normalized relative to the input image columns.
    /// The range of values is from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// The range of values is from 0 to 1.
    float y;
    /// width. width is normalized relative to the input image columns. The
    /// value range from 0 to 1.
    float width;
    /// height, height is normalized relative to the input image rows. The value
    /// range from 0 to 1.
    float height;
  };
  /// All objects, a vector of BoundingBox
  std::vector<BoundingBox> bboxes;
};

/**
 * @class MedicalDetectionPostProcess
 * @brief Class of the MedicalDetection post-process. It initializes the
 * parameters once instead of computing them every time when the program
 * execute.
 * */
class MedicalDetectionPostProcess {
 public:
  /**
   * @brief Create an MedicalDetectionPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The dpu model configuration information.
   * @return An unique pointer of MedicalDetectionPostProcess.
   */
  static std::unique_ptr<MedicalDetectionPostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config, int& real_batch_size);

  /**
   * @brief The post-processing function of the MedicalDetection network.
   * @return Struct of MedicalDetectionResult.
   */
  virtual MedicalDetectionResult medicaldetection_post_process(
      unsigned int idx) = 0;
  /**
   * @brief The batch mode post-processing function of the MedicalDetection
   * network.
   * @return The vector of struct of MedicalDetectionResult.
   */
  virtual std::vector<MedicalDetectionResult>
  medicaldetection_post_process() = 0;

  /**
   * @cond NOCOMMENTS
   */
  virtual ~MedicalDetectionPostProcess();

 protected:
  explicit MedicalDetectionPostProcess();
  MedicalDetectionPostProcess(const MedicalDetectionPostProcess&) = delete;
  MedicalDetectionPostProcess& operator=(const MedicalDetectionPostProcess&) =
      delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
