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
 * Filename: retinanet.hpp
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
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/nnpp/retinanet.hpp>

namespace xir {
    class Attrs;
};

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting position of vehicle, pedestrian, and so on.
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detection results, named RetinaNetResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_retinanet.jpg");
   auto retinanet = vitis::ai::RetinaNet::create("retinanet_traffic_pruned_0_9",true);
   auto results = retinanet->run(img);
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
 * @image latex images/sample_retinanet_result.jpg "detection result" width=\textwidth
 */
class RetinaNet : public ConfigurableDpuTaskBase {
public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * RetinaNet.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of RetinaNet class.
   *
   */
    static std::unique_ptr<RetinaNet> create(const std::string& model_name,
        bool need_preprocess = true);
  /**
   * @brief Factory function to get an instance of derived classes of class RetinaNet.
   *
   * @param model_name Model name
   * @param attrs Xir attributes
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of RetinaNet class.
   *
   */

    static std::unique_ptr<RetinaNet> create(const std::string& model_name,
        xir::Attrs* attrs, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
protected:
    explicit RetinaNet(const std::string& model_name, bool need_preprocess = true);
    explicit RetinaNet(const std::string& model_name, xir::Attrs* attrs, bool need_preprocess = true);
    RetinaNet(const RetinaNet&) = delete;

public:
    virtual ~RetinaNet();
  /**
   * @endcond
   */
public:
  /**
   * @brief Function to get running results of the RetinaNet neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return RetinaNetResult.
   *
   */
  //virtual vitis::ai::RetinaNetResult run(const cv::Mat& image) = 0;
   virtual std::vector<RetinaNetBoundingBox>run(const cv::Mat& image) = 0;

  /**
   * @brief Function to get running results of the RetinaNet neural network in
   * batch mode.
   *
   * @param images Input data of input images (vector<cv::Mat>).The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of RetinaNetResult.
   *
   */
  virtual std::vector<std::vector<vitis::ai::RetinaNetBoundingBox>> run(const std::vector<cv::Mat>& images) = 0;

};

}  // namespace ai
}  // namespace vitis
