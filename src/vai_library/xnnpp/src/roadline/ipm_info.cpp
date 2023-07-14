/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License,
 * Version 2.0 (the "License");
 * you may not use this file except in
 * compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by
 * applicable law or agreed to in writing, software
 * distributed under the
 * License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the
 * specific language governing permissions and
 * limitations under the
 * License.
 */

#include "ipm_info.hpp"

#include <cassert>
#include <iostream>

using namespace std;

namespace vitis {
namespace nnpp {
namespace roadline {

vector<int> IpmInfo::vx1;
vector<int> IpmInfo::vy1;

IpmInfo::IpmInfo(int ratio, float ipm_width, float ipm_height, float ipm_left,
                 float ipm_right, float ipm_top, float ipm_bottom,
                 float ipm_interpolation, float ipm_vp_portion,
                 float focal_length_x, float focal_length_y,
                 float optical_center_x, float optical_center_y,
                 float camera_height, float pitch, float yaw)
    : /* ratio_(ratio),  */ ipm_width_(ipm_width),
      ipm_height_(ipm_height),
      ipm_left_(ipm_left),
      ipm_right_(ipm_right),
      ipm_top_(ipm_top),
      ipm_bottom_(ipm_bottom),
      /* ipm_interpolation_(ipm_interpolation) ,*/
      /* ipm_vp_portion_(ipm_vp_portion) ,*/ focal_length_x_(focal_length_x),
      focal_length_y_(focal_length_y),
      optical_center_x_(optical_center_x),
      optical_center_y_(optical_center_y),
      camera_height_(camera_height),
      pitch_(pitch),
      yaw_(yaw) {
  initialize_ipm();
}

#if 0
// only for debug
void myprintv(vector<vector<float>> iv, std::string name) {
  std::cout << "------printing " << name << "-------\n";
  for(unsigned int i=0; i<iv.size(); i++) {
     for(unsigned int j=0; j<iv[0].size(); j++) {
         cout << iv[i][j] << " " ;
     }
     cout << "\n";
  }
}
#endif

IpmInfo::~IpmInfo() {}

void IpmInfo::IPM(const vector<int>& img, vector<int>& outImage) {
  for (int n = 0; n < ipm_height_ * ipm_width_; n++) {
    outImage[veci(n) * ipm_width_ + vecj(n)] =
        img[vy1[n] * ipm_width_ + vx1[n]];
  }
}

void IpmInfo::Recover(const vector<int>& img, vector<int>& outImage) {
  for (int n = 0; n < ipm_height_ * ipm_width_; n++) {
    outImage[vy1[n] * ipm_width_ + vx1[n]] =
        img[veci(n) * ipm_width_ + vecj(n)];
  }
}

int IpmInfo::initialize_ipm() {
  if (vx1.size() == ipm_height_ * ipm_width_ &&
      vy1.size() == ipm_height_ * ipm_width_) {
    // already initalized.
    return 0;
  }
  vector<vector<float>> vpp = GetVanishingPoint();

  vector<vector<float>> uvLimitsp{
      {(float)(int)vpp[0][0], (float)(int)ipm_right_, (float)(int)ipm_left_,
       (float)(int)vpp[0][0]},
      {(float)(int)ipm_top_, (float)(int)ipm_top_, (float)(int)ipm_top_,
       (float)(int)ipm_bottom_}};
  vector<vector<float>> xyLimits = TransformImage2Ground(uvLimitsp);
  auto xfMin = *std::min_element(xyLimits[0].begin(), xyLimits[0].end());
  auto xfMax = *std::max_element(xyLimits[0].begin(), xyLimits[0].end());
  auto yfMin = *std::min_element(xyLimits[1].begin(), xyLimits[1].end());
  auto yfMax = *std::max_element(xyLimits[1].begin(), xyLimits[1].end());

  auto stepRow = (yfMax - yfMin) / ipm_height_;
  auto stepCol = (xfMax - xfMin) / ipm_width_;
  vector<vector<float>> xyGrid(2, vector<float>(ipm_height_ * ipm_width_, 0));
  auto y = yfMax - 0.5 * stepRow;
  for (int i = 0; i < ipm_height_; i++) {
    auto x = xfMin + 0.5 * stepCol;
    for (int j = 0; j < ipm_width_; j++) {
      xyGrid[0][i * ipm_width_ + j] = x;
      xyGrid[1][i * ipm_width_ + j] = y;
      x += stepCol;
    }
    y -= stepRow;
  }

  auto uvGrid = TransformGround2Image(xyGrid);

  for (int i = 0; i < ipm_height_ * ipm_width_; i++) {
    if (uvGrid[0][i] <= ipm_left_ || uvGrid[0][i] >= ipm_right_)
      uvGrid[0][i] = 0;
    if (uvGrid[1][i] <= (int)ipm_top_ || uvGrid[1][i] >= (int)ipm_bottom_)
      uvGrid[1][i] = 0;
  }
  for (int i = 0; i < ipm_height_ * ipm_width_; i++) {
    vx1.push_back(int(uvGrid[0][i]));
    vy1.push_back(int(uvGrid[1][i]));
  }
  return 0;
}

vector<vector<float>> IpmInfo::TransformGround2Image(
    const vector<vector<float>>& iv) {
  vector<vector<float>> inPoints2{iv};
  inPoints2.erase(inPoints2.begin() + 2, inPoints2.end());
  vector<vector<float>> tmpV(1, vector<float>(inPoints2[0].size(), 1));
  vector<vector<float>> inPointsr3 = dot(tmpV, -camera_height_);
  inPoints2.insert(inPoints2.end(), inPointsr3.begin(), inPointsr3.end());

  float c1 = cos(pitch_ * PI / 180.0);
  float s1 = sin(pitch_ * PI / 180.0);
  float c2 = cos(yaw_ * PI / 180.0);
  float s2 = sin(yaw_ * PI / 180.0);

  vector<vector<float>> matp{
      {float((int)focal_length_x_ * c2 + c1 * s2 * (int)optical_center_x_),
       float(-(int)focal_length_x_ * s2 + c1 * c2 * (int)optical_center_x_),
       float(-s1 * (int)optical_center_x_)},
      {float(s2 * (-(int)focal_length_y_ * s1 + c1 * (int)optical_center_y_)),
       float(c2 * (-(int)focal_length_y_ * s1 + c1 * (int)optical_center_y_)),
       float(-(int)focal_length_y_ * c1 - s1 * (int)optical_center_y_)},
      {float(c1 * s2), float(c1 * c2), float(-s1)}};
  auto inPoints3 = dot(matp, inPoints2);
  for (unsigned int i = 0; i < inPoints3[2].size(); i++) {
    inPoints3[0][i] /= inPoints3[2][i];
    inPoints3[1][i] /= inPoints3[2][i];
  }

  inPoints3.erase(inPoints3.begin() + 2, inPoints3.end());
  return inPoints3;
}

vector<vector<float>> IpmInfo::TransformImage2Ground(
    const vector<vector<float>>& iv) {
  vector<vector<float>> inPoints3(iv);
  inPoints3.push_back(vector<float>(iv[0].size(), 1));
  auto c1 = cos(pitch_ * PI / 180.0);
  auto s1 = sin(pitch_ * PI / 180.0);
  auto c2 = cos(yaw_ * PI / 180.0);
  auto s2 = sin(yaw_ * PI / 180.0);

  vector<vector<float>> matp{
      {float(-camera_height_ * c2 / focal_length_x_),
       float(camera_height_ * s1 * s2 / focal_length_y_),
       float(camera_height_ * c2 * optical_center_x_ / focal_length_x_ -
             camera_height_ * s1 * s2 * optical_center_y_ / focal_length_y_ -
             camera_height_ * c1 * s2)},
      {float(camera_height_ * s2 / focal_length_x_),
       float(camera_height_ * s1 * c2 / focal_length_y_),
       float(-camera_height_ * s2 * optical_center_x_ / focal_length_x_ -
             camera_height_ * s1 * c2 * optical_center_y_ / focal_length_y_ -
             camera_height_ * c1 * c2)},
      {0, float(camera_height_ * c1 / focal_length_y_),
       float(-camera_height_ * c1 * optical_center_y_ / focal_length_y_ +
             camera_height_ * s1)},
      {0, float(-c1 / focal_length_y_),
       float(c1 * optical_center_y_ / focal_length_y_ - s1)}};

  auto inPoints4 = dot(matp, inPoints3);
  for (unsigned int i = 0; i < inPoints4[3].size(); i++) {
    inPoints4[0][i] /= inPoints4[3][i];
    inPoints4[1][i] /= inPoints4[3][i];
  }

  inPoints4.erase(inPoints4.begin() + 2, inPoints4.end());
  return inPoints4;
}

vector<vector<float>> IpmInfo::GetVanishingPoint() {
  vector<vector<float>> vpp{
      {float(sin(yaw_ * PI / 180.0) / cos(pitch_ * PI / 180.0))},
      {float(cos(yaw_ * PI / 180.0) / cos(pitch_ * PI / 180.0))},
      {0}};

  vector<vector<float>> tyawp{
      {float(cos(yaw_ * PI / 180.0)), float(-sin(yaw_ * PI / 180.0)), 0},
      {float(sin(yaw_ * PI / 180.0)), float(cos(yaw_ * PI / 180.0)), 0},
      {0, 0, 1}};

  vector<vector<float>> tpitchp{
      {1, 0, 0},
      {0, float(-sin(pitch_ * PI / 180.0)), float(-cos(pitch_ * PI / 180.0))},
      {0, float(cos(pitch_ * PI / 180.0)), float(-sin(pitch_ * PI / 180.0))}};

  auto transform = dot(tyawp, tpitchp);
  vector<vector<float>> t1p{
      {(float)(int)focal_length_x_, 0.0, (float)(int)optical_center_x_},
      {0, (float)(int)focal_length_y_, (float)(int)optical_center_y_},
      {0, 0, 1}};

  auto transform1 = dot(t1p, transform);
  auto transform2 = dot(transform1, vpp);
  return transform2;
}

template <typename T>
vector<vector<T>> IpmInfo::dot(const vector<vector<T>>& iv1, T i2) {
  vector<vector<T>> ov(iv1.size(), vector<T>(iv1[0].size(), 0));
  for (unsigned int i = 0; i < iv1.size(); i++) {
    for (unsigned int j = 0; j < iv1[0].size(); j++) {
      ov[i][j] = iv1[i][j] * i2;
    }
  }
  return ov;
}

template <typename T>
vector<vector<T>> IpmInfo::dot(const vector<vector<T>>& iv1,
                               const vector<vector<T>>& iv2) {
  assert((iv1.size() > 0) && (iv2.size() > 0) && (iv1[0].size() == iv2.size()));

  vector<vector<T>> ov(iv1.size(), vector<T>(iv2[0].size(), 0));
  for (unsigned int i = 0; i < ov.size(); i++) {
    for (unsigned int j = 0; j < ov[0].size(); j++) {
      for (unsigned int k = 0; k < iv1[0].size(); k++) {
        ov[i][j] += iv1[i][k] * iv2[k][j];
      }
    }
  }
  return ov;
}

}  // namespace roadline
}  // namespace nnpp
}  // namespace vitis
