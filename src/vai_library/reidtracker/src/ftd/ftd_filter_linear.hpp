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

#ifndef _FTD_FILTER_LINEAR_HPP_
#define _FTD_FILTER_LINEAR_HPP_
/// #define _FTD_DEBUG_
#include <array>
#include <opencv2/core.hpp>
#include <vector>

namespace vitis {
namespace ai {

// SpecifiedCfg: all_region, BGD
typedef std::tuple<std::array<int, 4>, std::array<int, 3>> SpecifiedCfg;

class FTD_Filter_Linear {
 public:
  FTD_Filter_Linear(){};
  ~FTD_Filter_Linear(){};
  void Init(const cv::Rect_<float> &bbox, int mode,
            const SpecifiedCfg &specifed_cfg);
  void UpdateDetect(const cv::Rect_<float> &bbox);
  void UpdateReidTracker(const cv::Rect_<float> &bbox);
  void UpdateFilter();
  cv::Rect_<float> GetPre();
  cv::Rect_<float> GetPost();

 private:
  void LeastSquare(std::vector<std::array<float, 2>> &coord,
                   std::array<float, 8> &para, float x, int region);
  void ClearSquare(std::vector<std::array<float, 2>> &coord,
                   std::array<float, 8> &para, float step);
  void LeastMean(std::vector<std::array<float, 2>> &coord,
                 std::array<float, 4> &para, float x, int region);
  void ClearMean(std::vector<std::array<float, 2>> &coord,
                 std::array<float, 4> &para, float step);
  std::vector<std::array<float, 2>> coordx;
  std::vector<std::array<float, 2>> coordy;
  std::vector<std::array<float, 2>> coords;
  std::vector<std::array<float, 2>> coordr;
  std::array<float, 8> parax;
  std::array<float, 8> paray;
  std::array<float, 8> paras;
  std::array<float, 4> parar;
  float frame_id=0.0;
  float frame_start=0.0;
  float frame_max=0.0;
  std::array<int, 4> allregion;
};

}  // namespace ai
}  // namespace vitis
#endif
