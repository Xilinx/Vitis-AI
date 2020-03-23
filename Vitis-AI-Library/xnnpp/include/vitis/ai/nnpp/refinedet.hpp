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
/*
 * Filename: refinedet.hpp
 *
 * Description:
 * This network is used to getting position and score of objects in the input
 * image Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace nnpp {
namespace refinedet {
class SSDdetector;
}
}  // namespace nnpp
}  // namespace vitis

namespace vitis {
namespace ai {
/**
 *@struct RefineDetResult
 *@brief Struct of the result with the refinedet network.
 *
 */
struct RefineDetResult {
  /// Width of the input image.
  int width;
  /// Height of the input image.
  int height;
  /**
   *@struct BoundingBox
   *@brief Struct of a object coordinate and confidence.
   */
  struct BoundingBox {
    /// x-coordinate , x is normalized relative to the input image cols ,the
    /// value range from 0 to 1.
    float x;
    /// y-coordinate , y is normalized relative to the input image rows ,the
    /// value range from 0 to 1.
    float y;
    /// body width , width is normalized relative to the input image cols , the
    /// value range from 0 to 1.
    float width;
    /// body height , heigth is normalized relative to the input image rows ,
    /// the value range from 0 to 1.
    float height;
    /// body detection confidence, the value range from 0 to 1.
    float score;
  };
  /// The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};

/**
 * @class RefineDetPostProcess
 * @brief Class of the refinedet post-process, it will initialize the parameters
 * once instead of compute them every time when the program execute.
 * */
class RefineDetPostProcess {
 public:
  /**
   * @brief Create an RefineDetPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   *   Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   *   Usage: output_tensors[output_index].
   * @param config The dpu model configuration information.
   * @return An unique printer of RefineDetPostProcess.
   */
  static std::unique_ptr<RefineDetPostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);

  /**
   * @brief Run refinedet post-process.
   * @return The struct of RefineDetResult.
   */
  virtual RefineDetResult refine_det_post_process(unsigned int idx) = 0;

  /**
   * @brief Run batch mode of refinedet post-process.
   * @return The vector of struct of RefineDetResult.
   */
  virtual std::vector<RefineDetResult> refine_det_post_process() = 0;
    /**
   * @cond NOCOMMENTS
   */
  virtual ~RefineDetPostProcess();

 protected:
  explicit RefineDetPostProcess();
  RefineDetPostProcess(const RefineDetPostProcess&) = delete;
  RefineDetPostProcess& operator=(const RefineDetPostProcess&) = delete;
    /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
