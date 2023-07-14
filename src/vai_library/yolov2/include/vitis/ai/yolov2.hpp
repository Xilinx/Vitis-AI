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
 * Filename: yolov2.hpp
 *
 * Description:
 * This network is used to detecting objects from an image, it will return
 * its coordinate, label and confidence.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/yolov2.hpp>
namespace vitis {
namespace ai {

/**
 *@brief Base class for detecting objects in the input image(cv::Mat).
 *Input is an image(cv::Mat).
 *Output is the position of the objects in the input image.
 *Sample code:
 *@code
   auto img = cv::imread("sample_yolov2.jpg");
   auto model = vitis::ai::YOLOv2::create("yolov2_voc");
   auto result = model->run(img);
   for (const auto &bbox : result.bboxes) {
     int label = bbox.label;
     float xmin = bbox.x * img.cols + 1;
     float ymin = bbox.y * img.rows + 1;
     float xmax = xmin + bbox.width * img.cols;
     float ymax = ymin + bbox.height * img.rows;
     if (xmax > img.cols)
       xmax = img.cols;
     if (ymax > img.rows)
       ymax = img.rows;
     float confidence = bbox.score;

     cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
     rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
              1, 0);
  }
  @endcode
 *
 */
class YOLOv2 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * YOLOv2.
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default
   *value is true.
   * @return An instance of YOLOv2 class.
   *
   */
  static std::unique_ptr<YOLOv2> create(const std::string& model_name,
                                        bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit YOLOv2();
  YOLOv2(const YOLOv2&) = delete;
  virtual ~YOLOv2();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the YOLOv2 neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return A Struct of YOLOv2Result.
   *
   */
  virtual YOLOv2Result run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running result of the YOLOv2 neural network
   * in batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of YOLOv2Result.
   *
   */
  virtual std::vector<YOLOv2Result> run(const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get InputWidth of the YOLOv2 network (input image columns).
   *
   * @return InputWidth of the YOLOv2 network
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the YOLOv2 network (input image rows).
   *
   *@return InputHeight of the YOLOv2 network.
   */
  virtual int getInputHeight() const = 0;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   * time.
   * @note Different DPU core the batch size may be different. This depends on
   * the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
