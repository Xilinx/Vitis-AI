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
#include <opencv2/core.hpp>
#include <xilinx/ai/nnpp/ssd.hpp>

namespace xilinx {
namespace ai {



/**
 * @brief Base class for detecting position of vehicle,pedestrian and so on.
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detection results, named SSDResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_ssd.jpg");
   auto ssd = xilinx::ai::SSD::create("ssd_traffic_pruned_0_9",true);
   auto results = ssd->run(img);
   for(const auto &r : results.bboxes){
      auto label = r.label;
      auto x = r.x * img.cols;
      auto y = r.y * img.rows;
      auto width = r.width * img.cols;
      auto heigth = r.height * img.rows;
      auto score = r.score;
      std::cout << "RESULT: " << label << "\t" << x << "\t" << y << "\t" <<
 width
         << "\t" << height << "\t" << score << std::endl;
   }
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_ssd_result.jpg "detection result" width=\textwidth

 */
class SSD {
public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * SSD.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of SSD class.
   *
   */
  static std::unique_ptr<SSD> create(const std::string &model_name,
                                     bool need_preprocess = true);

protected:
  explicit SSD();
  SSD(const SSD &) = delete;

public:
  virtual ~SSD();

public:
  /**
   * @brief Function of get result of the ssd neuron network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return SSDResult.
   *
   */
  virtual xilinx::ai::SSDResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get InputWidth of the SSD network (input image cols).
   *
   * @return InputWidth of the SSD network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeigth of the SSD network (input image rows).
   *
   *@return InputHeight of the SSD network.
   */

  virtual int getInputHeight() const = 0;
};
} // namespace ai
} // namespace xilinx
