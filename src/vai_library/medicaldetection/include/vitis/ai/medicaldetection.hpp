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
 * Filename: medicaldetection.hpp
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
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/nnpp/medicaldetection.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting five objects of Endoscopy Disease Detection 
 *        and Segmentation database (EDD2020) .
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detection results, named MedicalDetectionResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_medicaldetection.jpg");
   auto medicaldetection = vitis::ai::MedicalDetection::create("RefineDet-Medical_EDD_tf",true);
   auto results = medicaldetection->run(img);
   for(const auto &r : results.bboxes){
      auto label = r.label;
      auto x = r.x * img.cols;
      auto y = r.y * img.rows;
      auto width = r.width * img.cols;
      auto height = r.height * img.rows;
      auto score = r.score;
      std::cout << "RESULT: " << label << "\t" << x << "\t" << y << "\t" << width
         << "\t" << height << "\t" << score << std::endl;
   }
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_medicaldetection_result.jpg "detection result" width=\textwidth
 */
class MedicalDetection{
public:
  /**
   * @brief Factory function to get an instance of derived classes of class MedicalDetection.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of MedicalDetection class.
   *
   */
  static std::unique_ptr<MedicalDetection> create(const std::string &model_name,
                                     bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
protected:
  explicit MedicalDetection();
  MedicalDetection(const MedicalDetection &) = delete;

public:
  virtual ~MedicalDetection();
 /**
   * @endcond
   */

public:
  /**
   * @brief Function of get result of the MedicalDetection neural network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return MedicalDetectionResult.
   *
   */
  virtual vitis::ai::MedicalDetectionResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the MedicalDetection neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MedicalDetectionResult.
   *
   */
  virtual std::vector<vitis::ai::MedicalDetectionResult> run(const std::vector<cv::Mat> &imgs) = 0;

  /**
   * @brief Function to get InputWidth of the MedicalDetection network (input image cols).
   *
   * @return InputWidth of the MedicalDetection network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the MedicalDetection network (input image rows).
   *
   *@return InputHeight of the MedicalDetection network.
   */

  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   *@return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
} // namespace ai
} // namespace vitis
