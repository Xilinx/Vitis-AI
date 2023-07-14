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
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#pragma once
//#define BEV_SHOW_SCALE 8
#define BEV_SHOW_SCALE 4
#define BEV_SHOW_SIZE 128

using V1F = std::vector<float>;
using V1I = std::vector<int>;
using V2F = std::vector<V1F>;
using V2I = std::vector<V1I>;
using V3F = std::vector<V2F>;

std::tuple<V1F, V1F> get_dx_bx() {
  V2F bounds{{-51.2, 51.2, 0.8}, {-51.2, 51.2, 0.8}, {-10.0, 10.0, 20.0}};
  V1F bx(2);
  V1F dx(2);
  for (int i = 0; i < 2; i++) {
    bx[i] = bounds[i][2];
    dx[i] = bounds[i][0] + bounds[i][2] / 2.0;
  }
  // std::cout <<"bxdx :" << bx[0] << " " << bx[1] << "   " << dx[0] << " " <<
  // dx[1] <<"\n";
  return std::tuple<V1F, V1F>(bx, dx);
}

V3F einsum(const V3F& points, const V3F& rot_mat_T) {
  V3F ret(points.size(), V2F(points[0].size(), V1F(rot_mat_T[0].size(), 0)));
  for (unsigned int a = 0; a < points.size(); a++) {
    for (unsigned int i = 0; i < points[0].size(); i++) {
      for (unsigned int k = 0; k < rot_mat_T[0].size(); k++) {
        for (unsigned int j = 0; j < points[0][0].size(); j++) {
          ret[a][i][k] += points[a][i][j] * rot_mat_T[j][k][a];
        }
      }
    }
  }
  return ret;
}

V3F rotation_3d_in_axis(const V3F& points, const V1F& angles)  // only axis == 2
{
  V1F rot_sin(angles.size(), 0);
  V1F rot_sin_n(angles.size(), 0);
  V1F rot_cos(angles.size(), 0);
  for (unsigned int i = 0; i < angles.size(); i++) {
    rot_sin[i] = sin(angles[i]);
    rot_sin_n[i] = -rot_sin[i];
    rot_cos[i] = cos(angles[i]);
  }
  V3F rot_mat_T(3, V2F(3, V1F(points.size(), 0)));
  rot_mat_T[0][0] = rot_cos;
  rot_mat_T[0][1] = rot_sin_n;
  rot_mat_T[0][2] = V1F(points.size(), 0);
  rot_mat_T[1][0] = rot_sin;
  rot_mat_T[1][1] = rot_cos;
  rot_mat_T[1][2] = V1F(points.size(), 0);
  rot_mat_T[2][0] = V1F(points.size(), 0);
  rot_mat_T[2][1] = V1F(points.size(), 0);
  rot_mat_T[2][2] = V1F(points.size(), 1);

  return einsum(points, rot_mat_T);
}

float myfix_y(float in) { return (float)BEV_SHOW_SIZE - in; }

V3F get_corners(const std::vector<vitis::ai::CenterPointResult>& cr) {
  V1F angles;
  V2F dims;
  for (auto& it : cr) {
    angles.emplace_back(it.bbox[6]);
    dims.emplace_back(V1F{it.bbox[3], it.bbox[4], it.bbox[5]});
  };

  V2F cnorm{{-0.5, -0.5, 0.}, {-0.5, -0.5, 1.}, {-0.5, 0.5, 1.},
            {-0.5, 0.5, 0.},  {0.5, -0.5, 0.},  {0.5, -0.5, 1.},
            {0.5, 0.5, 1.},   {0.5, 0.5, 0.}};
  V3F points(cr.size(), V2F(cnorm.size(), V1F(3, 0)));
  for (int i = 0; i < (int)cr.size(); i++) {
    for (int j = 0; j < (int)cnorm.size(); j++) {
      for (int k = 0; k < (int)cnorm[0].size(); k++) {
        points[i][j][k] = dims[i][k] * cnorm[j][k];
      }
    }
  }
  // for(int i=0;i<8;i++) { std::cout << points[0][i][0] << " " <<
  // points[0][i][1] << " " << points[0][i][2] << "\n";}

  V3F ret = rotation_3d_in_axis(points, angles);
  // for(int i=0;i<8;i++) { std::cout << ret[0][i][0] << " " <<  ret[0][i][1] <<
  // " " << ret[0][i][2] << "\n";  }

  for (int i = 0; i < (int)ret.size(); i++) {
    for (int j = 0; j < (int)ret[0].size(); j++) {
      for (int k = 0; k < (int)ret[0][0].size(); k++) {
        ret[i][j][k] += cr[i].bbox[k];
      }
    }
  }
  // for(int i=0;i<8;i++) { std::cout << ret[0][i][0] << " " <<  ret[0][i][1] <<
  // " " << ret[0][i][2] << "\n"; }
  return ret;
}

cv::Mat draw_bev(const std::vector<vitis::ai::CenterPointResult>& cr) {
  /* index of result
      0 2   0   | 0 1   1
      2 6   0   | 1 3   1
      6 4   0   | 3 2   1
      4 0   0   | 2 0   1

    corner_full:
      -12.7943 -9.49712 -2.0719
      -12.7943 -9.49712 0.1786
      -7.27091 -10.1234 0.1786
      -7.27091 -10.1234 -2.0719
      -13.0351 -11.6206 -2.0719
      -13.0351 -11.6206 0.1786
      -7.51168 -12.2469 0.1786
      -7.51168 -12.2469 -2.0719
  */
  auto img =
      cv::Mat(BEV_SHOW_SIZE * BEV_SHOW_SCALE, BEV_SHOW_SIZE * BEV_SHOW_SCALE,
              CV_8UC3, cv::Scalar(255, 255, 255));
  V1F bx, dx;
  std::tie(dx, bx) = get_dx_bx();
  V2I line_idx{{0, 2}, {2, 6}, {6, 4}, {4, 0}};
  std::vector<cv::Scalar> vscalar = {
      cv::Scalar(0, 0, 255),  cv::Scalar(0, 255, 0),   cv::Scalar(0, 255, 255),
      cv::Scalar(255, 0, 0),  cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
      cv::Scalar(255, 64, 0), cv::Scalar(255, 0, 64),  cv::Scalar(64, 0, 255),
      cv::Scalar(64, 255, 0)};

  V3F corner_full = get_corners(cr);
  // format [n 8 3]  --> [ n, 4, 2]
  cv::Point pt1, pt2;
  for (int i = 0; i < (int)corner_full.size(); i++) {
    for (int j = 0; j < 4; j++) {
      pt1.x = ((corner_full[i][line_idx[j][0]][0] - bx[0]) / dx[0]) *
              BEV_SHOW_SCALE;
      pt1.y = myfix_y(((corner_full[i][line_idx[j][0]][1] - bx[1]) / dx[1])) *
              BEV_SHOW_SCALE;
      pt2.x = ((corner_full[i][line_idx[j][1]][0] - bx[0]) / dx[0]) *
              BEV_SHOW_SCALE;
      pt2.y = myfix_y(((corner_full[i][line_idx[j][1]][1] - bx[1]) / dx[1])) *
              BEV_SHOW_SCALE;

      cv::line(img, pt1, pt2, vscalar[cr[i].label]);
    }
  }

  // draw self
  cv::Point ptsx[4];
  const cv::Point* ppt[1] = {ptsx};
  int npts[] = {4};
  V2F pts{{-4.084 / 2. + 0.5, 1.85 / 2.},
          {4.084 / 2. + 0.5, 1.85 / 2.},
          {4.084 / 2. + 0.5, -1.85 / 2.},
          {-4.084 / 2. + 0.5, -1.85 / 2.}};
  for (int j = 0; j < 4; j++) {
    ptsx[j].y = ((pts[j][0] - bx[0]) / dx[0]) * BEV_SHOW_SCALE;
    ptsx[j].x = ((pts[j][1] - bx[1]) / dx[1]) * BEV_SHOW_SCALE;
  }
  cv::fillPoly(img, ppt, npts, 1, cv::Scalar(0x76, 0xb9, 0));
  return img;
}
