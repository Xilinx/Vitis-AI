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

#include "ftd_filter_linear.hpp"
#include <glog/logging.h>
#include <iostream>
#include <tuple>
using namespace std;
namespace vitis {
namespace ai {

cv::Rect_<float> ConvertBboxToZL(const cv::Rect_<float> &bbox) {
  cv::Rect_<float> z;
  z.x = bbox.x + bbox.width / 2.f;
  z.y = bbox.y + bbox.height / 2.f;
  z.width = bbox.width * bbox.height;
  z.height = bbox.width / bbox.height;
  return z;
}

cv::Rect_<float> ConvertZToBboxL(const cv::Rect_<float> &z) {
  if ((z.width > 0.f) && (z.height > 0.f)) {
    cv::Rect_<float> bbox;
    bbox.width = std::sqrt(z.width * z.height);
    bbox.height = z.width / bbox.width;
    bbox.x = z.x - bbox.width / 2.f;
    bbox.y = z.y - bbox.height / 2.f;
    return bbox;
  } else {
    return cv::Rect_<float>(0.f, 0.f, 0.f, 0.f);
  }
}

void FTD_Filter_Linear::LeastSquare(std::vector<std::array<float, 2>> &coord,
                                    std::array<float, 8> &para, float x,
                                    int region) {
  // if(!coord.empty()) std::cout<<"frameid: "<<frame_id<<" coords:
  // "<<coord.back()[0]<<std::endl;
  CHECK(coord.empty() || float(frame_id) > coord.back()[0])
      << "coord must be ascending";
  CHECK(region >= 1) << "error region";
  std::array<float, 2> tmp_coord;
  tmp_coord[0] = frame_id;
  tmp_coord[1] = x;
  para[2] += tmp_coord[0] * tmp_coord[1];
  para[3] += tmp_coord[0];
  para[4] += tmp_coord[0] * tmp_coord[0];
  para[5] += tmp_coord[1];
  para[6] += 1;
  para[7] += tmp_coord[1] * tmp_coord[1];
  coord.push_back(tmp_coord);
  while ((int)coord.size() > region) {
    para[2] -= coord[0][0] * coord[0][1];
    para[3] -= coord[0][0];
    para[4] -= coord[0][0] * coord[0][0];
    para[5] -= coord[0][1];
    para[6] -= 1;
    para[7] -= coord[0][1] * coord[0][1];
    coord.erase(coord.begin());
  }
  if (coord.size() == 1) {
    para[0] = 0.f;
    para[1] = tmp_coord[1];
  } else {
    float V = para[6] * para[4] - para[3] * para[3];
    para[0] = (para[6] * para[2] - para[3] * para[5]) / V;
    para[1] = (para[4] * para[5] - para[2] * para[3]) / V;
    // float A = para[2] - para[3]*para[5]/para[6];
    // float B = para[4] - para[3]*para[3]/para[6];
    // float C = para[7] - para[5]*para[5]/para[6];
    // float r = (A*A)/(B*C);
  }
}
void FTD_Filter_Linear::ClearSquare(std::vector<std::array<float, 2>> &coord,
                                    std::array<float, 8> &para, float step) {
  for (auto &tmp_coord : coord) {
    tmp_coord[0] -= step;
  }
  para[2] = para[2] - step * para[5];
  para[4] = para[4] - 2 * step * para[3] + step * step * para[6];
  para[3] = para[3] - step * para[6];
  if (coord.size() == 1) {
    para[0] = 0.f;
    para[1] = coord[0][1];
  } else {
    float V = para[6] * para[4] - para[3] * para[3];
    para[0] = (para[6] * para[2] - para[3] * para[5]) / V;
    para[1] = (para[4] * para[5] - para[2] * para[3]) / V;
  }
}

void FTD_Filter_Linear::LeastMean(std::vector<std::array<float, 2>> &coord,
                                  std::array<float, 4> &para, float x,
                                  int region) {
  CHECK(coord.empty() || float(frame_id) > coord.back()[0])
      << "coord must be ascending";
  CHECK(region >= 1) << "error region";
  std::array<float, 2> tmp_coord;
  tmp_coord[0] = frame_id;
  tmp_coord[1] = x;
  para[2] += tmp_coord[1];
  para[3] += 1;
  coord.push_back(tmp_coord);
  while ((int)coord.size() > region) {
    para[2] -= coord[0][1];
    para[3] -= 1;
    coord.erase(coord.begin());
  }
  if (coord.size() == 1) {
    para[0] = 0.f;
    para[1] = tmp_coord[1];
  } else {
    para[0] = 0.f;
    para[1] = para[2] / para[3];
  }
}
void FTD_Filter_Linear::ClearMean(std::vector<std::array<float, 2>> &coord,
                                  std::array<float, 4> &para, float step) {
  for (auto &tmp_coord : coord) {
    tmp_coord[0] -= step;
  }
  if (coord.size() == 1) {
    para[0] = 0.f;
    para[1] = coord[0][1];
  } else {
    para[0] = 0.f;
    para[1] = para[2] / para[3];
  }
}

void FTD_Filter_Linear::Init(const cv::Rect_<float> &bbox, int mode,
                             const SpecifiedCfg &specified_cfg) {
  switch (mode) {
    case 0: {
      LOG(FATAL) << "FTD_Filter_Linear Only support mode 1!";
    }; break;
    case 1: {
      frame_start = 0.0f;
      frame_max = 0.05f;
    }; break;
    default:
      break;
  }
  frame_id = frame_start;
  // allregion = std::get<0>(specified_cfg);
  allregion = std::array<int, 4>({3, 3, 1, 1});
  parax = std::array<float, 8>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  paray = std::array<float, 8>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  paras = std::array<float, 8>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  parar = std::array<float, 4>{0.f, 0.f, 0.f, 0.f};
  cv::Rect_<float> z = ConvertBboxToZL(bbox);
  LeastSquare(coordx, parax, z.x, allregion[0]);
  LeastSquare(coordy, paray, z.y, allregion[1]);
  LeastSquare(coords, paras, z.width, allregion[2]);
  LeastMean(coordr, parar, z.height, allregion[3]);
}

void FTD_Filter_Linear::UpdateDetect(const cv::Rect_<float> &bbox) {
  cv::Rect_<float> z = ConvertBboxToZL(bbox);
  LeastSquare(coordx, parax, z.x, allregion[0]);
  LeastSquare(coordy, paray, z.y, allregion[1]);
  LeastSquare(coords, paras, z.width, allregion[2]);
  LeastMean(coordr, parar, z.height, allregion[3]);
}

void FTD_Filter_Linear::UpdateReidTracker(const cv::Rect_<float> &bbox) {
  // cv::Rect_<float> z = ConvertBboxToZL(bbox);
  // LeastSquare(coordx, parax, z.x, allregion[0]);
  // LeastSquare(coordy, paray, z.y, allregion[1]);
  // LeastSquare(coords, paras, z.width, allregion[2]);
  // LeastMean(coordr, parar, z.height, allregion[3]);
}

void FTD_Filter_Linear::UpdateFilter() {}

cv::Rect_<float> FTD_Filter_Linear::GetPre() {
  cv::Rect_<float> z;
  frame_id += 0.001f;
  // change it when frame_id max
  if (frame_id >= frame_max) {
    frame_id -= (frame_max - frame_start);
    ClearSquare(coordx, parax, (frame_max - frame_start));
    ClearSquare(coordy, paray, (frame_max - frame_start));
    ClearSquare(coords, paras, (frame_max - frame_start));
    ClearMean(coordr, parar, (frame_max - frame_start));
  }
  z.x = parax[0] * frame_id + parax[1];
  z.y = paray[0] * frame_id + paray[1];
  z.width = paras[0] * frame_id + paras[1];
  z.height = parar[0] * frame_id + parar[1];
  return ConvertZToBboxL(z);
}

cv::Rect_<float> FTD_Filter_Linear::GetPost() {
  cv::Rect_<float> z;
  z.x = parax[0] * frame_id + parax[1];
  z.y = paray[0] * frame_id + paray[1];
  z.width = paras[0] * frame_id + paras[1];
  z.height = parar[0] * frame_id + parar[1];
  return ConvertZToBboxL(z);
}

}  // namespace ai
}  // namespace vitis
