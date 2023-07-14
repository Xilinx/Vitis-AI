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

 * Filename: solo.hpp
 *
 * Description:
 * This network is used to detecting featuare from a input image.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details.
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/nnpp/solo.hpp>

#include "vitis/ai/configurable_dpu_task.hpp"
namespace vitis {
namespace ai {
/**
 * @brief Base class for SOLO semantic segmentation from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is the enlarged image.
 *
 * @note The input image size is 640x360
 *
 * Sample code:
 * @code
  if (argc < 2) {
    cerr << "usage: " << argv[0] << "  modelname  image_file_url " << endl;
    abort();
  }
  Mat input_img = imread(argv[2]);
  if (input_img.empty()) {
    cerr << "can't load image! " << argv[2] << endl;
    return -1;
  }
  auto det = vitis::ai::Solo::create(argv[1]);
  Mat ret_img = det->run(input_img).feat;
  imwrite("sample_solo_result.png", ret_img);
    @endcode
 */
class Solo : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Solo.
   *
   *@param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default
   * value is true.
   * @return An instance of Solo class.
   *
   */
  static std::unique_ptr<Solo> create(const std::string& model_name,
                                      bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit Solo(const std::string& model_name, bool need_preprocess);
  Solo(const Solo&) = delete;
  virtual ~Solo();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the SOLO neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return SoloResult.
   *
   */
  virtual SoloResult run(const cv::Mat& image) = 0;

  /**
   * @brief Function to get running result of the SOLO neural network in batch
   * mode.
   *
   * @param images Input data of input images (vector<cv::Mat>).
   *
   * @return vector of SoloResult.
   *
   */
  virtual std::vector<SoloResult> run(const std::vector<cv::Mat>& images) = 0;

};
}  // namespace ai
}  // namespace vitis
