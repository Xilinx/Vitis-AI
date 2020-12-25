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
 * Filename: pointpillars.hpp
 *
 * Description:
 * This network is used to detecting objects from a input points data.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

using V1F = std::vector<float>;
using V2F = std::vector<V1F>;
using V1I=std::vector<int>;
using V2I=std::vector<V1I>;

enum e_flag{
  E_RGB=0x01,
  E_BEV=0x02
};

/**
 * @struct DISPLAY_PARAM
 * @brief  Four data structure get from the calibration information. 
 *  It is mainly used for accuracy test or bev image drawing.
 *  See detail in readme in the overview for more information.
 */
struct DISPLAY_PARAM{
 /// P2 information: 
 V2F P2;
 /// rect information
 V2F rect;
 /// Trv2c information
 V2F Trv2c;
 /// p2rect information
 V2F p2rect;
};

/**
 * @struct ANNORET 
 * @brief Struct of the result returned by the pointpillars neuron network in the annotation mode.
 *  It is mainly used for accuracy test or bev image drawing
 */
struct ANNORET{
  /// Name of detected result in vector: such as Car Cylist Pedestrian.
  std::vector<std::string> name;
  /// Label of detected result.
  V1I label;
  /// Truncated information.
  V1F truncated;
  /// Occluded information.
  V1I occluded;
  /// Alpha information. 
  V1F alpha;
  /// bbox information.
  V2I bbox;
  /// Dimensions information.
  V2F dimensions;
  /// Location information.
  V2F location;
  /// rotation_y information.
  V1F rotation_y;
  /// Score information.
  V1F score;
  /// box3d_camera information.
  V2F box3d_camera;
  /// box3d_lidar information.
  V2F box3d_lidar;
  /// Inner function to clear all fields.
  void clear() {
      name.clear();
      label.clear();
      truncated.clear();
      occluded.clear();
      alpha.clear();
      bbox.clear();
      dimensions.clear();
      location.clear();
      rotation_y.clear();
      score.clear();
      box3d_camera.clear();
      box3d_lidar.clear();
  }
};

/**
 * @struct PPResult
 * @brief Struct of the final result returned by the pointpillars neuron network.
 */
struct PPResult{
  /// Final box predicted.
  V2F final_box_preds;
  /// Final scores predicted.
  V1F final_scores;
  /// Final label predicted.
  V1I label_preds;
};

/**
 * @struct PointPillarsResult
 * @brief Struct of the final result returned by the pointpillars 
 *  neuron network encapsulated with width/height information.
 */
struct PointPillarsResult {
  /// Width of network input.
  int width=0;
  /// Height of network input.
  int height=0;
  /// Final result returned by the pointpillars neuron network.
  PPResult ppresult;
};

/**
 * @brief Base class for pointpillars .
 *
 * Input is points data.
 *
 * Output is a struct of detection results, named PointPillarsResult.
 *
 * Sample code :
   @code
     ...
     auto net = vitis::ai::PointPillars::create("pointpillars_kitti_12000_0_pt", "pointpillars_kitti_12000_1_pt", true);
     V1F PointCloud ;
     int len = getfloatfilelen( lidar_path);
     PointCloud.resize( len );
     myreadfile(PointCloud.data(), len, lidar_path);
     auto res = net->run(PointCloud);
     ...
   please see the test sample for detail.
   @endcode
 *
 */
class PointPillars {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * PointPillars.
   *
   * @param model_name  Model name for PointNet
   * @param model_name1 Model name for RPN
   * @return An instance of PointPillars class.
   *
   */
  static std::unique_ptr<PointPillars> create(
      const std::string &model_name, const std::string& model_name1);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit PointPillars();
  PointPillars(const PointPillars &) = delete;

 public:
  virtual ~PointPillars();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the PointPillars neuron network.
   *
   * @param v1f: point data in vector<float>
   *
   * @return PointPillarsResult.
   *
   */
  virtual vitis::ai::PointPillarsResult run( const V1F& v1f) = 0;
  /**
   * @brief Function to get input batch of the PointPillars network.
   *
   * @return Input batch of the PointPillars network.
   */
  virtual size_t get_input_batch() const = 0;

  /**
   * @brief Function to produce the visible result from PointPillarsResult after calling run().
   *        This is a helper function which can be ignored if you wants 
   *        to process the PointPillarsResult using another method.
   *
   * @param res: [input] PointPillarsResult from run().
   * @param flag: [input] which visible result to produce. can be assigned to below values:
   *              E_BEV        : only produce BEV picture
   *              E_RGB        : only produce RGB picture
   *              E_BEV|E_RGB  : produce both pictures 
   * @param dispp: [input] display parameter for the Points data.
   *               Refer to the readme in the overview for more detail.
   * @param rgb_map: [input|output] : original rgb picture for drawing detect result. 
   *               It can be blank (cv::Mat{}), if only BEV is required
   * @param bev_map: [input|output] original bev picture for drawing detect result. 
   *               It can be blank (cv::Mat{}), if only RGB required
   * @param width: [input] original rgb picture width.
   * @param height: [input] original rgb picture height.
   * @param annoret: [output] return the annoret variable for accuracy calculation.
   */
  virtual void do_pointpillar_display(PointPillarsResult& res, int flag,  DISPLAY_PARAM& dispp,
            cv::Mat& rgb_map, cv::Mat& bev_map, int width, int height, ANNORET& annoret) = 0;
};

}  // namespace ai
}  // namespace vitis
