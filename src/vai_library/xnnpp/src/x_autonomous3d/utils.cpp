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
#include "utils.hpp"
#include <string.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "float.h"

using namespace std;
namespace vitis {
namespace ai {
namespace x_autonomous3d {

void sigmoid_n(std::vector<float>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = (1. / (1. + exp(-src[i])));
  }
  src.swap(dst);
}

void iou_quality_cal(std::vector<float>& src) {
  for (size_t i = 0; i < src.size(); ++i) {
    auto val = (src[i] + 1.0) * 0.5;
    if (val > 1.0) {
      val = 1.0;
    } else if (val < 0.000001) {
      val = 0.0;
    }
    src[i] = val;
  }
}

void iou_quality_cal(int8_t* output_tensor_ptr, float scale, int size,
                     std::vector<float>& dst) {
  for (auto i = 0; i < size; ++i) {
    auto val = ((output_tensor_ptr[i] * scale) + 1.0) * 0.5;
    if (val > 1.0) {
      val = 1.0;
    } else if (val < 0.000001) {
      val = 0.0;
    }
    dst[i] = val;
  }
}

void heatmap_calculate_with_iou_quality(std::vector<float>& heatmap,
                                        const std::vector<float>& iou_quality,
                                        const std::vector<float>& iou_alpha,
                                        int class_num) {
  int batch = iou_quality.size();
  for (auto b = 0; b < batch; ++b) {
    for (auto cls_idx = 0; cls_idx < class_num; ++cls_idx) {
      auto idx = b * class_num + cls_idx;
      auto sigmoid_val = (1. / (1. + exp(-heatmap[idx])));
      heatmap[idx] = std::pow(sigmoid_val, (1.0 - iou_alpha[cls_idx])) *
                     std::pow(iou_quality[b], iou_alpha[cls_idx]);
    }
  }
}

void heatmap_calculate_with_iou_quality(int8_t* hm_tensor_ptr,
                                        float hm_tensor_scale,
                                        const std::vector<float>& iou_quality,
                                        const std::vector<float>& iou_alpha,
                                        int class_num, vector<float>& dst) {
  int batch = iou_quality.size();
  for (auto b = 0; b < batch; ++b) {
    for (auto cls_idx = 0; cls_idx < class_num; ++cls_idx) {
      auto idx = b * class_num + cls_idx;
      auto sigmoid_val =
          (1. / (1. + exp(-(hm_tensor_ptr[idx] * hm_tensor_scale))));
      dst[idx] = std::pow(sigmoid_val, (1.0 - iou_alpha[cls_idx])) *
                 std::pow(iou_quality[b], iou_alpha[cls_idx]);
    }
  }
}

int topk(vector<ScoreIndex> input, int k) {
  int size = input.size();
  k = size < k ? size : k;
  std::make_heap(input.begin(), input.begin() + k, ScoreIndex::compare);
  for (int i = k; i < size; ++i) {
    if (ScoreIndex::compare(input[i], input[0])) {
      std::pop_heap(input.begin(), input.begin() + k, ScoreIndex::compare);
      input[k - 1] = input[i];
      std::push_heap(input.begin(), input.begin() + k, ScoreIndex::compare);
    }
  }
  return k;
}

void exp_n(std::vector<float>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = exp(src[i]);
  }
  src.swap(dst);
}

void atan2_n(std::vector<float>& dst, const std::vector<float>& src) {
  auto size = dst.size();
  for (size_t i = 0; i < size; ++i) {
    dst[i] = atan2(src[2 * i], src[2 * i + 1]);
  }
}

float cross(const Point& a, const Point& b) { return a.x * b.y - a.y * b.x; }

float cross(const Point& p1, const Point& p2, const Point& p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

int check_rect_cross(const Point& p1, const Point& p2, const Point& q1,
                     const Point& q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

int check_in_box2d(const float* box, const Point& p) {
  // params: (7) [x, y, z, dx, dy, dz, heading]
  const float MARGIN = 1e-2;

  float center_x = box[0], center_y = box[1];
  float angle_cos = cos(-box[6]),
        angle_sin =
            sin(-box[6]);  // rotate the point in the opposite direction of box
  float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
  float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

  return (fabs(rot_x) < box[3] / 2 + MARGIN &&
          fabs(rot_y) < box[4] / 2 + MARGIN);
}

int intersection(const Point& p1, const Point& p0, const Point& q1,
                 const Point& q0, Point& ans) {
  // fast exclusion
  if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

  // check cross standing
  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

  // calculate intersection of two lines
  float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > EPS) {
    ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;

    ans.x = (b0 * c1 - b1 * c0) / D;
    ans.y = (a1 * c0 - a0 * c1) / D;
  }

  return 1;
}

void rotate_around_center(const Point& center, const float angle_cos,
                          const float angle_sin, Point& p) {
  float new_x =
      (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
  float new_y =
      (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p.set(new_x, new_y);
}

int point_cmp(const Point& a, const Point& b, const Point& center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

float box_overlap(const float* box_a, const float* box_b) {
  // params box_a: [x, y, z, dx, dy, dz, heading]
  // params box_b: [x, y, z, dx, dy, dz, heading]

  float a_angle = box_a[6], b_angle = box_b[6];
  float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2,
        a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
  float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
  float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
  float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
  float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

  Point center_a(box_a[0], box_a[1]);
  Point center_b(box_b[0], box_b[1]);

  Point box_a_corners[5];
  box_a_corners[0].set(a_x1, a_y1);
  box_a_corners[1].set(a_x2, a_y1);
  box_a_corners[2].set(a_x2, a_y2);
  box_a_corners[3].set(a_x1, a_y2);

  Point box_b_corners[5];
  box_b_corners[0].set(b_x1, b_y1);
  box_b_corners[1].set(b_x2, b_y1);
  box_b_corners[2].set(b_x2, b_y2);
  box_b_corners[3].set(b_x1, b_y2);

  // get oriented corners
  float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
  float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

  for (int k = 0; k < 4; k++) {
    rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
    rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
  }

  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  // get intersection of lines
  Point cross_points[16];
  Point poly_center;
  int cnt = 0, flag = 0;

  poly_center.set(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  // check corners
  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(box_a, box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(box_b, box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  // sort the points of polygon
  Point temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

  // get the overlap areas
  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }

  return fabs(area) / 2.0;
}

float iou_bev(const float* box_a, const float* box_b) {
  // params box_a: [x, y, z, dx, dy, dz, heading]
  //     // params box_b: [x, y, z, dx, dy, dz, heading]
  float sa = box_a[3] * box_a[4];
  float sb = box_b[3] * box_b[4];
  float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
              const float nms, const float conf, vector<size_t>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  stable_sort(order.begin(), order.end(),
              [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
                return ls.first > rs.first;
              });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  int nms_cnt = 0;
  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.f;
      /*
      if (ENV_PARAM(DEBUG_NMS_USE_2D)) {
        //ovr = cal_iou(boxes[j], boxes[i]);
      } else {
      */
      ovr = iou_bev(boxes[j].data(), boxes[i].data());
      /*
      }
      */
      nms_cnt++;
      /*
            if (ENV_PARAM(DEBUG_NMS)) {
              if (ENV_PARAM(DEBUG_NMS_USE_2D)) {
                std::cout << "2D ";
              } else {
                std::cout << "3D ";
              }
              std::cout << "iou of top 1000: " << i
                        << " and top 1000: " << j
                        << " iou is : " << ovr << std::endl;
            }
      */
      if (ovr >= nms) exist_box[j] = false;
    }
  }

  /*
    LOG_IF(INFO, ENV_PARAM(DEBUG_NMS_IOU))
          << "iou cnt: " << nms_cnt;
  */
}
}  // namespace x_autonomous3d
}  // namespace ai
}  // namespace vitis
