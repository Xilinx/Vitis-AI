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
 * Filename: roadline.hpp
 *
 * Description:
 * This network is used to detecting road line
 *
 * Please refer to document "Xilnx_AI_SDK_User_Guide.pdf" for more details of
 *these APIs.
 */

#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {
/**
 * @struct RoadLineResult
 * @brief Struct of the result returned by the roadline network.
 */
struct RoadLineResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /**
   *@struct Line
   *@brief Struct of the result returned by the roadline network.
   */
  struct Line {
    /// road line type, the value range from 0 to 3.
    /// \li \c 0 : background
    /// \li \c 1 : white dotted line
    /// \li \c 2 : white solid line
    /// \li \c 3 : yollow line
    int type;
    /// point clusters, make line from these.
    std::vector<cv::Point> points_cluster;
  };
  /// the vector of line
  std::vector<Line> lines;
};

/**
 * @class RoadLinePostProcess
 * @brief Class of the roadline post-process, it will initialize the parameters
 *once instead of compute them every time when the program execute.
 * */
class RoadLinePostProcess {
 public:
  /**
   * @brief Create an RoadLinePostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   *   Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   *  Usage: output_tensors[output_index].
   * @param config The dpu model configuration information.
   * @return An unique printer of RoadLinePostProcess.
   */
  static std::unique_ptr<RoadLinePostProcess> create(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);

  /**
   * @brief Run roadline post-process.
   * @return The struct of RoadLineResult.
   */
  virtual RoadLineResult road_line_post_process(int inWidth, int inHeight, unsigned int idx) = 0;
  /**
   * @brief Run roadline post-process in batch mode.
   * @return The vector of struct of RoadLineResult.
   */
  virtual std::vector<RoadLineResult> road_line_post_process(const std::vector<int>& inWidth, const std::vector<int>& inHeight) = 0;
   /**
   * @cond NOCOMMENTS
   */ 
  virtual ~RoadLinePostProcess();

 protected:
  explicit RoadLinePostProcess();
  RoadLinePostProcess(const RoadLinePostProcess&) = delete;
  RoadLinePostProcess& operator=(const RoadLinePostProcess&) = delete;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
