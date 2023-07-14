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
 * Filename: clocs.hpp
 *
 * Description:
 * This class is used for 3D object detection in autonomous driving.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace vitis {
namespace ai {

namespace clocs {

/**
 * @struct ClocsInfo
 * @brief Structure to describe clocs input information
 */
struct ClocsInfo {
  /// P2 size: 16
  std::vector<float> calib_P2;
  /// Tr_velo_to_cam size: 16
  std::vector<float> calib_Trv2c;
  /// R0_rect size: 16
  std::vector<float> calib_rect;
  /// 3D Lidar Points.
  std::vector<float> points;
  /// 2D Image
  cv::Mat image;
};

}  // namespace clocs

struct ClocsResult {
  struct PPBbox {
    /// Confidence
    float score;
    /// 3D lidar bounding box: x, y, z, x-size, y-size, z-size, yaw.
    std::vector<float> bbox;
    /// Classification, for Clocs, only one class: Car.
    uint32_t label;
  };

  /// All bounding boxes
  std::vector<PPBbox> bboxes;
};

/**
 * @brief Base class for clocs.
 *
 * Input is points data and related params.
 *
 * Output is a struct of detection results, named ClocsResult.
 *
 * Sample code :
   @code
     ...
     std::string yolo_model_name = "clocs_yolox_pt";
     std::string pp_model_0 = "clocs_pointpillars_kitti_0_pt";
     std::string pp_model_1 = "clocs_pointpillars_kitti_1_pt";
     std::string fusion_model_name = "clocs_fusion_cnn_pt";

     auto clocs = vitis::ai::Clocs::create(yolo_model_name, pp_model_0,
 pp_model_1, fusion_model_name, true);


     vector<ClocsInfo> batch_clocs_info(input_num);
     // see the test sample to read ClocsInfo
     //
     auto batch_ret = clocs->run(batch_clocs_info);

     ...
   please see the test sample for detail.
   @endcode
 */
class Clocs {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Clocs
   *
   * @param yolo The yolo model name
   * @param pointpillars_0 The pointpillars 0 model name
   * @param pointpillars_1 The pointpillars 1 model name
   * @param fusionnet The funsion model name
   * @param need_preprocess  Normalize with mean/scale or not, default
   *  value is true.
   * @return An instance of ClocsPointPillars class.
   */
  static std::unique_ptr<Clocs> create(const std::string& yolo,
                                       const std::string& pointpillars_0,
                                       const std::string& pointpillars_1,
                                       const std::string& fusionnet,
                                       bool need_preprocess = true);

  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Clocs();
  Clocs(const Clocs&) = delete;
  Clocs& operator=(const Clocs&) = delete;

 public:
  virtual ~Clocs();
  /**
   * @endcond
   */
  /**
   * @brief Function to get input width of the first model of
   * Clocs class.
   *
   * @return Input width of the first model.
   */
  virtual int getInputWidth() const = 0;

  /**
   * @brief Function to get input height of the first model of
   * Clocs class
   *
   * @return Input height of the first model.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of inputs processed by the DPU at one
   * time.
   * @note Batch size of different DPU core may be different, it depends on the
   * IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;

  /**
   * @brief Function to get the points dim of an input points. The dim depends
   * on the last channel of the first model of Clocs.
   *
   * @return The dim of points.
   */
  virtual int getPointsDim() const = 0;

  /**
   * @brief Function of get result of the Clocs class.
   *
   * @param input Clocs input info.
   *
   * @return ClocsResult.
   *
   */
  virtual ClocsResult run(const clocs::ClocsInfo& input) = 0;
  /**
   * @brief Function of get result of the Clocs class with prepared 2d result,
   * This api is only for debug.
   *
   * @param detect2d_result preloaded 2d result.
   *
   * @param input  Clocs input info.
   *
   * @return ClocsResult.
   *
   */

  virtual ClocsResult run(const std::vector<float>& detect2d_result,
                          const clocs::ClocsInfo& input) = 0;

  /**
   * @brief Function of get result of the Clocs class in
   * batch mode.
   *
   * @param batch_inputs Clocs input infos.
   *
   * @return The vector of ClocsResult.
   *
   */
  virtual std::vector<ClocsResult> run(
      const std::vector<clocs::ClocsInfo>& batch_inputs) = 0;
  /**
   * @brief Function of get result of the Clocs class in
   * batch mode. This api is only for debug.
   *
   * @param batch_detect2d_result preloaded 2d results.
   *
   * @param batch_inputs Clocs input infos.
   *
   * @return The vector of ClocsResult.
   *
   */

  virtual std::vector<ClocsResult> run(
      const std::vector<std::vector<float>>& batch_detect2d_result,
      const std::vector<clocs::ClocsInfo>& batch_inputs) = 0;
};

}  // namespace ai
}  // namespace vitis
