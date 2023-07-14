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

 * Filename: reid.hpp
 *
 * Description:
 * This network is used to detecting featuare from a input image.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details.
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/nnpp/reid.hpp>

#include "vitis/ai/configurable_dpu_task.hpp"
namespace vitis {
namespace ai {
/**
 * @brief Base class for getting feat from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output image reid feat.
 *
 * Sample code:
 * @code
  if(argc < 3){
      cerr<<"need two images"<<endl;
      return -1;
  }
  Mat imgx = imread(argv[1]);
  if(imgx.empty()){
      cerr<<"can't load image! "<<argv[1]<<endl;
      return -1;
  }
  Mat imgy = imread(argv[2]);
  if(imgy.empty()){
      cerr<<"can't load image! "<<argv[2]<<endl;
      return -1;
  }
  auto det = vitis::ai::Reid::create("reid");
  Mat featx = det->run(imgx).feat;
  Mat featy = det->run(imgy).feat;
  double dismat= cosine_distance(featx, featy);
  printf("dismat : %.3lf \n", dismat);
    @endcode
 *
 */
class Reid : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Reid.
   *
   *@param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default
   * value is true.
   * @return An instance of Reid class.
   *
   */
  static std::unique_ptr<Reid> create(const std::string& model_name,
                                      bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit Reid(const std::string& model_name, bool need_preprocess);
  Reid(const Reid&) = delete;
  virtual ~Reid();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the ReID neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return ReidResult.
   *
   */
  virtual ReidResult run(const cv::Mat& image) = 0;

  /**
   * @brief Function to get running result of the ReID neural network in batch
   * mode.
   *
   * @param images Input data of input images (vector<cv::Mat>).
   *
   * @return vector of ReidResult.
   *
   */
  virtual std::vector<ReidResult> run(const std::vector<cv::Mat>& images) = 0;

  /**
   * @brief Function to get running results of the reid neural network in
   * batch mode , used to receive user's xrt_bo to support zero copy.
   *
   * @param input_bos The vector of vart::xrt_bo_t.
   *
   * @return The vector of ReidResult.
   *
   */
  virtual std::vector<ReidResult> run(
      const std::vector<vart::xrt_bo_t>& input_bos) = 0;
};
}  // namespace ai
}  // namespace vitis
