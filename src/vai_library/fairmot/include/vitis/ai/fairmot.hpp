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

 * Filename: fairmot.hpp
 *
 * Description:
 * This network is used to detecting featuare from a input image.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details.
 * details of these APIs.
 */
#pragma once
#include "vitis/ai/configurable_dpu_task.hpp"
namespace vitis {
namespace ai {
/**
 * @brief Base class for detecting persons and feats from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is the enlarged image.
 *
 * @note The input image size is 640x480
 *
 * Sample code:
 * @code
  auto image_file = string(argv[2]);
  Mat input_img = imread(image_file);
  if (input_img.empty()) {
    cerr << "can't load image! " << argv[2] << endl;
    return -1;
  }
  auto det = vitis::ai::FairMot::create(argv[1]);
  auto result = det->run(input_img);
  auto feats = result.feats;
  auto bboxes = result.bboxes;
  auto img = input_img.clone();
  for (auto i = 0u; i < bboxes.size(); ++i) {
    auto box = bboxes[i];
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

    LOG(INFO) << "RESULT " << box.label << " :\t" << xmin << "\t" << ymin
              << "\t" << xmax << "\t" << ymax << "\t" << score << "\n";
    LOG(INFO) << "feat size: " << feats[i].size()
              << " First 5 digits: " << feats[i].data[0] + 0.0f << " "
              << feats[i].data[1] + 0.0f << " " << feats[i].data[2] + 0.0f
              << " " << feats[i].data[3] + 0.0f << " "
              << feats[i].data[4] + 0.0f << endl;
    cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  cv::Scalar(0, 255, 0), 1, 1, 0);
  }
  auto out = image_file.substr(0, image_file.size() - 4) + "_out.jpg";
  LOG(INFO) << "write result to " << out;
  cv::imwrite(out, img);
      @endcode
 *
 * Display of the model results:
 * @image latex images/sample_fairmot_result.jpg "result image" width=300px
 *
 */

/**
 *@struct BoundingBox
 *@brief Struct of an object coordinates and confidence.
 */
struct BoundingBox {
  /// x-coordinate. x is normalized relative to the input image columns.
  /// Range from 0 to 1.
  float x;
  /// y-coordinate. y is normalized relative to the input image rows.
  /// Range from 0 to 1.
  float y;
  /// Body width. Width is normalized relative to the input image columns,
  /// Range from 0 to 1.
  float width;
  /// Body height. Heigth is normalized relative to the input image rows,
  /// Range from 0 to 1.
  float height;
  /// Body detection label. The value ranges from 0 to 21.
  int label;
  /// Body detection confidence. The value ranges from 0 to 1.
  float score;
};

/**
 * @brief Result with the Rcan network.
 */
struct FairMotResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// The vector of reid feat.
  std::vector<cv::Mat> feats;
  /// The vector of BoundingBox.
  std::vector<BoundingBox> bboxes;
};

class FairMot : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * FairMot.
   *
   *@param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default
   * value is true.
   * @return An instance of FairMot class.
   *
   */
  static std::unique_ptr<FairMot> create(const std::string& model_name,
                                         bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit FairMot(const std::string& model_name, bool need_preprocess);
  FairMot(const FairMot&) = delete;
  virtual ~FairMot();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the FAIRMOT neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return FairMotResult.
   *
   */
  virtual FairMotResult run(const cv::Mat& image) = 0;

  /**
   * @brief Function to get running result of the FAIRMOT neural network in
   * batch mode.
   *
   * @param images Input data of input images (vector<cv::Mat>).
   *
   * @return vector of FairMotResult.
   *
   */
  virtual std::vector<FairMotResult> run(
      const std::vector<cv::Mat>& images) = 0;
};
}  // namespace ai
}  // namespace vitis
