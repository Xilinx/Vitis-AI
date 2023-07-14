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
 * Filename: clocs_pointpillars.hpp
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

namespace xir {
class Attrs;
};

namespace vitis {
namespace ai {

struct ClocsPointPillarsResult {
  struct PPBbox {
    /// Confidence
    float score;
    /// Bounding box: x, y, z, x-size, y-size, z-size, yaw, custom value and so
    /// on.
    std::vector<float> bbox;
    /// Classification
    uint32_t label;
  };

  /// All bounding boxes
  std::vector<PPBbox> bboxes;
};

struct ClocsPointPillarsMiddleResult {
  std::vector<float> scores;
  std::vector<std::vector<float>> bboxes;
  std::vector<int> labels;
};

/**
 * @brief Base class for clocs_pointpillars .
 *
 * Input is points data and related params.
 *
 * Output is a struct of detection results, named ClocsPointPillarsResult.
 *
 * Sample code :
   @code
     ...
     std::string anno_file_name = "./sample_pointpillars_nus.info";
     PointsInfo points_info;
     std::string model_0 = "clocs_pointpillars_40000_64_0_pt";
     std::string model_1 = "clocs_pointpillars_40000_64_1_pt";
     auto pointpillars = vitis::ai::ClocsPointPillars::create(
          model_0, model_1);
     auto points_dim = pointpillars->getPointsDim();
     read_inno_file_pp_nus(anno_file_name, points_info, points_dim,
 points_info.sweep_infos);

     auto ret = pointpillars->run(points_info);

     ...
   please see the test sample for detail.
   @endcode
 */
class ClocsPointPillars {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * ClocsPointPillars
   *
   * @param model_name_0  The first model name
   * @param model_name_1  The second model name
   * @param need_preprocess  Normalize with mean/scale or not, default
   *  value is true.
   * @return An instance of ClocsPointPillars class.
   */
  static std::unique_ptr<ClocsPointPillars> create(
      const std::string& model_name_0, const std::string& model_name_1,
      bool need_preprocess = true);

  /**
   * @brief Factory function to get an instance of derived classes of class
   * ClocsPointPillars
   *
   * @param model_name_0  The first model name
   * @param model_name_1  The second model name
   * @param attrs XIR attributes, used to bind different models to the same dpu
   * core
   * @param need_preprocess  Normalize with mean/scale or not, default
   *  value is true.
   * @return An instance of ClocsPointPillars class.
   */

  static std::unique_ptr<ClocsPointPillars> create(
      const std::string& model_name_0, const std::string& model_name_1,
      xir::Attrs* attrs, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit ClocsPointPillars();
  ClocsPointPillars(const ClocsPointPillars&) = delete;
  ClocsPointPillars& operator=(const ClocsPointPillars&) = delete;

 public:
  virtual ~ClocsPointPillars();
  /**
   * @endcond
   */
  /**
   * @brief Function to get input width of the first model of
   * ClocsPointPillars class.
   *
   * @return Input width of the first model.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get input height of the first model of
   *ClocsPointPillars class
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
   * on the last channel of the first model of ClocsPointPillars network.
   *
   * @return The dim of points.
   */
  virtual int getPointsDim() const = 0;

  /**
   * @brief Function of get result of the ClocsPointPillars neural network.
   *
   * @param input_points Filtered points data.
   *
   * @return ClocsPointPillarsResult.
   *
   */

  virtual void setMultiThread(bool val) = 0;

  virtual ClocsPointPillarsResult run(
      const std::vector<float>& input_points) = 0;
  /**
   * @brief Function of get result of the ClocsPointPillars neural network in
   * batch mode.
   *
   * @param batch_points Filtered points data in batch mode.
   *
   * @return The vector of ClocsPointPillarsResult.
   *
   */

  virtual std::vector<ClocsPointPillarsResult> run(
      const std::vector<std::vector<float>>& batch_input_points) = 0;
};

}  // namespace ai
}  // namespace vitis
