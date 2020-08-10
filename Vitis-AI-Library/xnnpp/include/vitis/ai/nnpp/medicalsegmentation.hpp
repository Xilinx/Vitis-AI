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
 * Filename: ssd.hpp
 *
 * Description:
 * This network is used to detecting objects from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <vitis/ai/proto/dpu_model_param.pb.h>
// #include <vart/dpu/dpu_runner.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {
/**
 * @struct MedicalSegmentationResult
 * @brief Struct of the result returned by the ssd neuron network.
 */
struct MedicalSegmentationResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;

  std::vector<cv::Mat> segmentation;
};

/**
 * @class MedicalSegmentationPostProcess
 * @brief Class of the ssd post-process, it will initialize the parameters once
 * instead of compute them every time when the program execute.
 * */
class MedicalSegmentationPostProcess {
 public:
  /**
   * @brief Create an MedicalSegmentationPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The dpu model configuration information.
   * @return An unique printer of MedicalSegmentationPostProcess.
   */
  static std::unique_ptr<MedicalSegmentationPostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);

  /**
   * @brief The post-processing function of the ssd network.
   * @return The struct of MedicalSegmentationResult.
   */
  virtual MedicalSegmentationResult medicalsegmentation_post_process(unsigned int idx) = 0;
  /**
   * @brief The batch mode post-processing function of the ssd network.
   * @return The vector of struct of MedicalSegmentationResult.
   */
  virtual std::vector<MedicalSegmentationResult> medicalsegmentation_post_process() = 0;

    /**
   * @cond NOCOMMENTS
   */
  virtual ~MedicalSegmentationPostProcess();

 protected:
  explicit MedicalSegmentationPostProcess();
  MedicalSegmentationPostProcess(const MedicalSegmentationPostProcess&) = delete;
  MedicalSegmentationPostProcess& operator=(const MedicalSegmentationPostProcess&) = delete;
    /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
