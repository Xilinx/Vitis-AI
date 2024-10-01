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
 * Filename: YOLOv7.hpp
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
#include <vitis/ai/nnpp/yolov7.hpp>
namespace vitis {
namespace ai {

class YOLOv7 : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * YOLOv7.
   *
   * @param model_name Model name
   *
   * @param need_preprocess Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of YOLOv7 class.
   *
   */
  static std::unique_ptr<YOLOv7> create(const std::string& model_name,
                                        bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit YOLOv7(const std::string& model_name, bool need_preprocess);
  YOLOv7(const YOLOv7&) = delete;

 public:
  virtual ~YOLOv7();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the YOLOv7 neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return YOLOv7Result.
   *
   */
  virtual YOLOv7Result run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running result of the YOLOv7 neural network
   * in batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of YOLOv7Result.
   *
   */
  virtual std::vector<YOLOv7Result> run(const std::vector<cv::Mat>& images) = 0;
};
}  // namespace ai
}  // namespace vitis
