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

namespace xir {
class Attrs;
};

namespace vitis {
namespace ai {

using V1F = std::vector<float>;
using V2F = std::vector<V1F>;
using V3F = std::vector<V2F>;
using V1I = std::vector<int>;
using V2I = std::vector<V1I>;
using V3I = std::vector<V2I>;

namespace fusion_cnn {

struct DetectResult {
  struct BoundingBox {
    float score;
    V1F bbox;
  };
  std::vector<BoundingBox> bboxes;
};

struct FusionParam {
  V2F p2;
  V2F rect;
  V2F trv2c;
  int img_width;
  int img_height;
};

}  // namespace fusion_cnn

class FusionCNN {
 public:
  static std::unique_ptr<FusionCNN> create(const std::string& model_name,
                                           bool need_preprocess = true);

  static std::unique_ptr<FusionCNN> create(const std::string& model_name,
                                           xir::Attrs* attrs,
                                           bool need_preprocess = true);

  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit FusionCNN();
  FusionCNN(const FusionCNN&) = delete;
  FusionCNN& operator=(const FusionCNN&) = delete;

 public:
  virtual ~FusionCNN();
  /**
   * @endcond
   */
  /**
   * @brief Function to get input width of the first model of
   * FusionCNN class.
   *
   * @return Input width of the first model.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get input height of the first model of
   *FusionCNN class
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
   * @brief Function of get result of the FusionCNN neural network.
   *
   * @param input_points Filtered points data.
   *
   * @return Fusion_CnnResult.
   *
   */
  virtual vitis::ai::fusion_cnn::DetectResult run(
      const vitis::ai::fusion_cnn::DetectResult& detect_result_2d,
      vitis::ai::fusion_cnn::DetectResult& detect_result_3d,
      const vitis::ai::fusion_cnn::FusionParam& fusion_param) = 0;

  virtual std::vector<fusion_cnn::DetectResult> run(
      const std::vector<fusion_cnn::DetectResult>& batch_detect_result_2d,
      std::vector<fusion_cnn::DetectResult>& batch_detect_result_3d,
      const std::vector<fusion_cnn::FusionParam>& batch_fusion_params) = 0;
};

}  // namespace ai
}  // namespace vitis
