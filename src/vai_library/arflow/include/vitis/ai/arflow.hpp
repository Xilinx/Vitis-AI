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
 * Filename: ARFlow.hpp
 *
 * Description:
 * SA-Gate is a neural network that is used for indoor segmentation.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details
 * of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for ARFlow.
 *
 * Input is a pair images which are RGB image (cv::Mat) and HHA map generated
 * with depth map (cv::Mat).
 *
 * Output is a heatmap where each pixels is predicted with a semantic category,
 * like chair, bed, usual object in indoor.
 *
 * Sample code:
 * @code
    Mat img1 = cv::imread("sample_arflow1.jpg");
    Mat img2 = cv::imread("sample_arflow2.jpg");
    auto segmentation = vitis::ai::ARFlow::create("ARFlow-master-Q2", true);
    auto result = segmentation->run(img_bgr, img_hha);
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_RGBDsegmentation_result.jpg "out image"
 width=\textwidth
 */
class ARFlow : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * ARFlow.
   *
   * @param model_name Model name
   *
   * @param need_preprocess Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of ARFlow class.
   *
   */
  static std::unique_ptr<ARFlow> create(const std::string& model_name,
                                        bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit ARFlow(const std::string& model_name, bool need_preprocess);
  ARFlow(const ARFlow&) = delete;

 public:
  virtual ~ARFlow();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the ARFlow neural
   * network.
   *
   * @param image_bgr Input data of input image (cv::Mat).
   * @param image_hha Input data of input image_hha (cv::Mat).
   *
   * @return SegmentationResult.
   *
   */
  virtual std::vector<vitis::ai::library::OutputTensor> run(
      const cv::Mat& image_1, const cv::Mat& image_2) = 0;
  virtual std::vector<vitis::ai::library::OutputTensor> run(
      const std::vector<cv::Mat>& image_1,
      const std::vector<cv::Mat>& image_2) = 0;
};
}  // namespace ai
}  // namespace vitis
