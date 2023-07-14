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
 * Filename: vehicleclassification.hpp
 *
 * Description:
 * This network is used to getting vehicleclassification in the input image
 * Please refer to document "Xilinx_AI_SDK_User_guide.pdf" for more details of
 * these APIs.
 */

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/nnpp/vehicleclassification.hpp>

#include "vitis/ai/configurable_dpu_task.hpp"
namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting objects in the input image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is index and score of objects in the input image.
 *
 * Sample code:
 * @code
   auto image = cv::imread("sample_vehicleclassification.jpg");
   auto network = vitis::ai::VehicleClassification::create(
                  "vehicle_type_resnet18_pt");
   auto result = network->run(image);
   for (const auto &r : result.scores) {
      auto score = r.score;
      auto index = result.lookup(r.index);
   }
   @endcode
 *
 */

class VehicleClassification : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * VehicleClassification.
   *
   * @param model_name Model name.
   * @param need_preprocess Normalize with mean/scale or not,
   *default value is true.
   *
   * @return An instance of VehicleClassification class.
   *
   */
  static std::unique_ptr<VehicleClassification> create(
      const std::string& model_name, bool need_preprocess = true);

 public:
  /**
   * @cond NOCOMMENTS
   */
  explicit VehicleClassification(const std::string& model_name,
                                 bool need_preprocess);
  VehicleClassification(const VehicleClassification&) = delete;
  virtual ~VehicleClassification();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running results of the vehicleclassification neural
   * network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return VehicleClassificationResult.
   *
   */
  virtual vitis::ai::VehicleClassificationResult run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the vehicleclassification neural
   * network in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>). The size
   * of input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of VehicleClassificationResult.
   *
   */
  virtual std::vector<vitis::ai::VehicleClassificationResult> run(
      const std::vector<cv::Mat>& images) = 0;
};

}  // namespace ai
}  // namespace vitis
