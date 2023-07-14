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
#include <opencv2/core.hpp>
#include <vector>

namespace vitis { namespace ai { namespace pp {

using V1F = std::vector<float>;
using V2F = std::vector<V1F>;
using V1I = std::vector<int>;
using V2I = std::vector<V1I>;

enum e_flag { E_RGB = 0x01, E_BEV = 0x02 };

/**
 * @struct DISPLAY_PARAM
 * @brief  Four data structure getting from the calibration information.
 *  It is mainly used for accuracy test or bev image drawing.
 *  See detail in the overview/samples/pointpillars/readme for more information.
 */
struct DISPLAY_PARAM {
  /// P2 information
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
 * @brief Struct of the result returned by the pointpillars neural network in
 * the annotation mode. It is mainly used for accuracy test or bev image drawing
 */
struct ANNORET {
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
 * @brief Struct of the final result returned by the pointpillars neural
 * network.
 */
struct PPResult {
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
 *  neural network encapsulated with width/height information.
 */
struct PointPillarsResult {
  /// Width of network input.
  int width = 0;
  /// Height of network input.
  int height = 0;
  /// Final result returned by the pointpillars neural network.
  PPResult ppresult;
};


}}}

