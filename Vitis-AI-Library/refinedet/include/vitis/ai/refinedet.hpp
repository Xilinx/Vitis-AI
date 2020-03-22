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
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/refinedet.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting pedestrian in the input image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is position and score of pedestrian in the input image.
 *
 * Sample code:
 * @code
  auto det = vitis::ai::RefineDet::create("refinedet_pruned_0_8");
  auto image = cv::imread("sample_refinedet.jpg");
  cout << "load image" << endl;
  if (image.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }

  auto results = det->run(image);

  auto img = image.clone();
  for (auto &box : results.bboxes) {
      float x = box.x * (img.cols);
      float y = box.y * (img.rows);
      int xmin = x;
      int ymin = y;
      int xmax = x + (box.width) * (img.cols);
      int ymax = y + (box.height) * (img.rows);
      float score = box.score;
      xmin = std::min(std::max(xmin, 0), img.cols);
      xmax = std::min(std::max(xmax, 0), img.cols);
      ymin = std::min(std::max(ymin, 0), img.rows);
      ymax = std::min(std::max(ymax, 0), img.rows);

      cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(0, 255, 0), 1, 1, 0);
  }
  auto out = "sample_refinedet_result.jpg";
  LOG(INFO) << "write result to " << out;
  cv::imwrite(out, img);
  @endcode
 *
 *  Display of the model results:
 *  @image latex images/sample_refinedet_result.jpg " result image" width=\textwidth
 *
 */
class RefineDet {
public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * RefineDet.
   *
   * @param model_name
   * @param need_preprocess Normalize with mean/scale or not,
   *default value is true.
   * @return An instance of RefineDet class.
   *
   */
  static std::unique_ptr<RefineDet> create(const std::string &model_name,
                                           bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
public:
  explicit RefineDet();
  RefineDet(const RefineDet &) = delete;
  virtual ~RefineDet();
  /**
   * @endcond
   */
public:
  /**
   * @brief Function to get running result of the RefineDet neuron network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return A Struct of RefineDetResult.
   *
   */

  virtual RefineDetResult run(const cv::Mat &image) = 0;
  /**
   * @brief Function to get running result of the RefineDet neuron network in batch mode.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return vector of Struct of RefineDetResult.
   *
   */
  virtual std::vector<RefineDetResult> run(const std::vector<cv::Mat> &images) = 0;
  /**
   * @brief Function to get InputWidth of the refinedet network (input image
   * cols).
   *
   * @return InputWidth of the refinedet network
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeigth of the refinedet network (input image
   *rows).
   *
   *@return InputHeight of the refinedet network.
   */
  virtual int getInputHeight() const = 0;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   *@return Batch size.
   */
  virtual size_t get_input_batch() const = 0;

};
} // namespace ai
} // namespace vitis
