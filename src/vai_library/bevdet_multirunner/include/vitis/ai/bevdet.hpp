/*
 * Copyright 2019 xilinx Inc.
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
 * Filename: BEVdet.hpp
 *
 * Description:
 * This network is used to detecting objects from an image, it will return
 * its coordinate, label and confidence.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
namespace vitis {
namespace ai {

struct CenterPointResult {
  float bbox[9];
  /// Bounding box 3d: {x, y, z, x_size, y_size, z_size, yaw,vel1,vel2}
  float score;
  /// Score
  uint32_t label;
  //'car',         'truck',   'construction_vehicle',
  //'bus',         'trailer', 'barrier',
  //'motorcycle',  'bicycle', 'pedestrian',
  //'traffic_cone'
};

/**
 *@brief Base class for detecting objects in the input image(cv::Mat).
 *Input is an image(cv::Mat).
 *Output is the position of the objects in the input image.
 *Sample code:
 *@code
 TODO
  std::vector<cv::Mat> images;
  for (auto name : image_names) {
    images.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
  }
  auto model = vitis::ai::BEVdet::create(C2D2_lite_0_pt, C2D2_lite_1_pt);
  auto result = model->run(images);
  std::cout << result;
  @endcode
 *
 */
class BEVdet {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * BEVdet.
   * @param model_name Model name
   * @param use_aie Whether to use aie to accelerate , default value is false.
   * @return An instance of BEVdet class.
   *
   */
  static std::unique_ptr<BEVdet> create(const std::string& model_name,
                                        bool use_aie = false);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit BEVdet();
  BEVdet(const BEVdet&) = delete;
  virtual ~BEVdet();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the BEVdet neural network.
   *
   * @param images Input data of input images (std::vector<cv::Mat>).
   * @param input_bins Input data of input bins
   * (std::vector<std::vector<char>>).
   *
   * @return A std::vector<CenterPointResult> data.
   *
   */
  virtual std::vector<CenterPointResult> run(
      const std::vector<cv::Mat>& images,
      const std::vector<std::vector<char>>& input_bins) = 0;
};
}  // namespace ai
}  // namespace vitis
