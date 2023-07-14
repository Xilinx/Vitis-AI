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
 * Filename: efficientdet_d2.hpp
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
#include <vector>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/nnpp/ssd.hpp>

namespace vitis {
namespace ai {
/**
 * @struct EfficientDetD2Result
 * @brief Struct of the result returned by the EfficientDetD2 neural network.
 */

// using EfficientDetD2Result = vitis::ai::SSDResult;
struct EfficientDetD2Result {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /**
   * @struct BoundingBox
   * @brief  Struct of an object coordinate, confidence and classification.
   */
  struct BoundingBox {
    /// Classification
    int label;
    /// Confidence
    float score;
    /// x-coordinate. x is normalized relative to the input image columns.
    /// Range from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// Range from 0 to 1.
    float y;
    /// Width. Width is normalized relative to the input image columns,
    /// Range from 0 to 1.
    float width;
    /// Height. Heigth is normalized relative to the input image rows,
    /// Range from 0 to 1.
    float height;
  };
  /// All objects, a vector of BoundingBox
  std::vector<BoundingBox> bboxes;
};

/**
 * @class EfficientDetD2PostProcess
 * @brief Class of the EfficientDetD2 post-process. It initializes the
 * parameters once instead of computing them each time the program executes.
 * */
class EfficientDetD2PostProcess {
 public:
  /**
   * @brief Create an EfficientDetD2PostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The DPU model configuration information.
   * @return An unique pointer of EfficientDetD2PostProcess.
   */
  static std::unique_ptr<EfficientDetD2PostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);

  /**
   * @brief The batch mode post-processing function of the EfficientDetD2
   * network.
   * @param batch_size num of batch input
   * @param image_scales image scale to fit the network input size
   * @param swidths batch input image widths
   * @param sheights batch input image heights
   * @return The vector of struct of EfficientDetD2Result.
   */
  virtual std::vector<EfficientDetD2Result> postprocess(
      size_t batch_size, const std::vector<int>& swidths,
      const std::vector<int>& sheights,
      const std::vector<float>& image_scales) = 0;
  /**
   * @cond NOCOMMENTS
   */
  virtual ~EfficientDetD2PostProcess();

 protected:
  explicit EfficientDetD2PostProcess();
  EfficientDetD2PostProcess(const EfficientDetD2PostProcess&) = delete;
  EfficientDetD2PostProcess& operator=(const EfficientDetD2PostProcess&) =
      delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
