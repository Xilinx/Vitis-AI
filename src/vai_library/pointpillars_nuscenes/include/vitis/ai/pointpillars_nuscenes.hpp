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
 * Filename: pointpillars_nuscenes.hpp
 *
 * Description:
 * This network is used to detecting objects from a input points data.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/nnpp/pointpillars_nuscenes.hpp>

namespace xir {
class Attrs;
};

namespace vitis {
namespace ai {

namespace pointpillars_nus {

/**
 * @struct CamInfo
 * @brief Camera information of input points
 */
struct CamInfo {
  /// Timestamp of input points.
  uint64_t timestamp=0;
  /// Sensor to lidar translation params.
  std::array<float, 3> s2l_t;
  /// Sensor to lidar rotation params.
  std::array<float, 9> s2l_r;
  /// Camera intrinsic params.
  std::array<float, 9> cam_intr;
};

/**
 * @struct Points
 * @brief Structure to describe input points data
 */
struct Points {
  /// Points dim.
  int dim=0;
  /// Points data.
  std::shared_ptr<std::vector<float>> points;
};

/**
 * @struct SweepInfo
 * @brief Structure to describe sweeps
 */
struct SweepInfo {
  /// Camera information.
  CamInfo cam_info;
  /// Points.
  Points points;
};

/**
 * @struct PointsInfo
 * @brief Structure to describe points information
 */
struct PointsInfo {
  /// Camera information.
  std::vector<CamInfo> cam_info;
  /// Points.
  Points points;
  /// Timestamp of points.
  uint64_t timestamp=0;
  /// Sweeps information.
  std::vector<SweepInfo> sweep_infos;
};

}  // namespace pointpillars_nus

/**
 * @brief Base class for pointpillars_nuscenes .
 *
 * Input is points data and related params.
 *
 * Output is a struct of detection results, named PointPillarsNuscenesResult.
 *
 * Sample code :
   @code
     ...
     std::string anno_file_name = "./sample_pointpillars_nus.info";
     PointsInfo points_info;
     std::string model_0 = "pointpillars_nuscenes_40000_64_0_pt";
     std::string model_1 = "pointpillars_nuscenes_40000_64_1_pt";
     auto pointpillars = vitis::ai::PointPillarsNuscenes::create(
          model_0, model_1);
     auto points_dim = pointpillars->getPointsDim();
     read_inno_file_pp_nus(anno_file_name, points_info, points_dim,
 points_info.sweep_infos);

     auto ret = pointpillars->run(points_info);

     ...
   please see the test sample for detail.
   @endcode
 */
class PointPillarsNuscenes {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * PointPillarsNuscenes
   *
   * @param model_name_0  The first model name
   * @param model_name_1  The second model name
   * @param need_preprocess  Normalize with mean/scale or not, default
   *  value is true.
   * @return An instance of PointPillarsNuscenes class.
   */
  static std::unique_ptr<PointPillarsNuscenes> create(
      const std::string& model_name_0, const std::string& model_name_1,
      bool need_preprocess = true);

  /**
   * @brief Factory function to get an instance of derived classes of class
   * PointPillarsNuscenes
   *
   * @param model_name_0  The first model name
   * @param model_name_1  The second model name
   * @param attrs XIR attributes, used to bind different models to the same dpu
   * core
   * @param need_preprocess  Normalize with mean/scale or not, default
   *  value is true.
   * @return An instance of PointPillarsNuscenes class.
   */

  static std::unique_ptr<PointPillarsNuscenes> create(
      const std::string& model_name_0, const std::string& model_name_1,
      xir::Attrs* attrs, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit PointPillarsNuscenes();
  PointPillarsNuscenes(const PointPillarsNuscenes&) = delete;
  PointPillarsNuscenes& operator=(const PointPillarsNuscenes&) = delete;

 public:
  virtual ~PointPillarsNuscenes();
  /**
   * @endcond
   */
  /**
   * @brief Function to get input width of the first model of
   * PointPillarsNuscenes class.
   *
   * @return Input width of the first model.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get input height of the first model of
   *PointPillarsNuscenes class
   *
   *@return Input height of the first model.
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
   * on the last channel of the first model of PointPillarsNuscenes network.
   *
   * @return The dim of points.
   */
  virtual int getPointsDim() const = 0;

  /**
   * @brief Function to get filtered sweeps points data in batch mode.
   * @param input An object of structure PointsInfo, include points data and
   * other params.
   * @return The vector of filtered points.
   */
  virtual std::vector<float> sweepsFusionFilter(
      const vitis::ai::pointpillars_nus::PointsInfo& input) = 0;

  /**
   * @brief Function to get filtered sweep points data.
   * @param batch_input Vector of PointsInfo, the size of batch_input should be
   * equal to batch num.
   *
   * @return Filtered points.
   */
  virtual std::vector<std::vector<float>> sweepsFusionFilter(
      const std::vector<vitis::ai::pointpillars_nus::PointsInfo>&
          batch_input) = 0;
  /**
   * @brief Function of get result of the PointPillarsNuscenes neural network.
   *
   * @param input_points Filtered points data.
   *
   * @return PointPillarsNuscenesResult.
   *
   */
  virtual PointPillarsNuscenesResult run(
      const std::vector<float>& input_points) = 0;

  /**
   * @brief Function of get result of the PointPillarsNuscenes neural network in
   * batch mode.
   *
   * @param batch_points Filtered points data in batch mode.
   *
   * @return The vector of PointPillarsNuscenesResult.
   *
   */
  virtual std::vector<PointPillarsNuscenesResult> run(
      const std::vector<std::vector<float>>& batch_points) = 0;

  /**
   * @brief Function of get result of the PointPillarsNuscenes neural network.
   *
   * @param input An object of structure PointsInfo, include points data and
   * other params.
   *
   * @return PointPillarsNuscenesResult.
   *
   */
  virtual PointPillarsNuscenesResult run(
      const vitis::ai::pointpillars_nus::PointsInfo& input) = 0;

  /**
   * @brief Function of get result of the PointPillarsNuscenes neural network.
   *
   * @param batch_input Vector of PointsInfo, the size of batch_input should be
   * equal to batch num.
   *
   * @return The vector of PointPillarsNuscenesResult.
   *
   */
  virtual std::vector<PointPillarsNuscenesResult> run(
      const std::vector<vitis::ai::pointpillars_nus::PointsInfo>&
          batch_input) = 0;
};

}  // namespace ai
}  // namespace vitis
