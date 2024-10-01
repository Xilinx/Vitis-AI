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
 * Filename: ofa_yolo.hpp
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
#include <vitis/ai/nnpp/ofa_yolo.hpp>
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
 *
  auto yolo = vitis::ai::OFAYOLO::create("ofa_yolo_pt", true);

  Mat img = cv::imread("sample_ofa_yolo.jpg");
  auto results = yolo->run(img);

  for (auto& box : results.bboxes) {
    int label = box.label;
    float xmin = box.x * img.cols + 1;
    float ymin = box.y * img.rows + 1;
    float xmax = xmin + box.width * img.cols;
    float ymax = ymin + box.height * img.rows;
    if (xmin < 0.) xmin = 1.;
    if (ymin < 0.) ymin = 1.;
    if (xmax > img.cols) xmax = img.cols;
    if (ymax > img.rows) ymax = img.rows;
    float confidence = box.score;

    cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
              1, 0);
  }
  imwrite("result.jpg", img);

   @endcode
 *
 */
class OFAYOLO : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * OFAYOLO.
   *
   * @param model_name Model name
   *
   * @param need_preprocess Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of OFAYOLO class.
   *
   */
  static std::unique_ptr<OFAYOLO> create(const std::string& model_name,
                                        bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit OFAYOLO(const std::string& model_name, bool need_preprocess);
  OFAYOLO(const OFAYOLO&) = delete;

 public:
  virtual ~OFAYOLO();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the OFA_YOLO neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return OFAYOLOResult.
   *
   */
  virtual OFAYOLOResult run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running result of the OFA_YOLO neural network
   * in batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of OFAYOLOResult.
   *
   */
  virtual std::vector<OFAYOLOResult> run(const std::vector<cv::Mat>& images) = 0;
};
}  // namespace ai
}  // namespace vitis
