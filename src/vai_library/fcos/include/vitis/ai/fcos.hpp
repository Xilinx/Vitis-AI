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
 * Filename: fcos.hpp
 *
 * Description:
 * This network is used to detecting object from an image, it will return
 * its coordinate, label and confidence.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details
 * of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/nnpp/fcos.hpp>
namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting objects in the input image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is the position of the pedestrians in the input image.
 *
 * Sample code:
 * @code
    static cv::Scalar getColor(int label) {
      return cv::Scalar(label * 2, 255 - label * 2, label + 50);
    }

    auto yolo = vitis::ai::FCOS::create("fcos", true);

    Mat img = cv::imread("sample_fcos.jpg");
    auto results = fcos->run(img);
    for (const auto& result : results.bboxes) {
      int label = result.label;
      auto& box = result.box;
      LOG_IF(INFO, is_jpeg) << "RESULT: " << label << "\t" << std::fixed
                            << std::setprecision(2) << box[0] << "\t" << box[1]
                            << "\t" << box[2] << "\t" << box[3] << "\t"
                            << std::setprecision(6) << result.score << "\n";

      cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]),
                    getColor(label), 1, 1, 0);
    }

   @endcode
 *
 */

class FCOS : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * FCOS.
   *
   * @param model_name Model name
   *
   * @param need_preprocess Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of FCOS class.
   *
   */
  static std::unique_ptr<FCOS> create(const std::string& model_name,
                                        bool need_preprocess = true);
  /**
   * @brief Factory function to get an instance of derived classes of class
   * FCOS.
   *
   * @param model_name Model name
   * @param attrs XIR attributes, used to bind different models to the same dpu
   * core
   * @param need_preprocess Normalize with mean/scale or not, default
   * value is true.
   *
   * @return An instance of FCOS class.
   *
   */
  static std::unique_ptr<FCOS> create(const std::string& model_name,
                                        xir::Attrs* attrs,
                                        bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit FCOS(const std::string& model_name, bool need_preprocess);
  explicit FCOS(const std::string& model_name, xir::Attrs* attrs,
                  bool need_preprocess);
  FCOS(const FCOS&) = delete;

 public:
  virtual ~FCOS();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the FCOS neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return FCOSResult.
   *
   */
  virtual FCOSResult run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running result of the FCOS neural network
   * in batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of FCOSResult.
   *
   */
  virtual std::vector<FCOSResult> run(const std::vector<cv::Mat>& images) = 0;
};
}  // namespace ai
}  // namespace vitis
