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

#include "vitis/ai/nnpp/apply_nms.hpp"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float box_c(vector<float> a, vector<float> b) {
  float top = min(a[1] - a[3] / 2.0, b[1] - b[3] / 2.0);
  float bot = max(a[1] + a[3] / 2.0, b[1] + b[3] / 2.0);
  float left = min(a[0] - a[2] / 2.0, b[0] - b[2] / 2.0);
  float right = max(a[0] + a[2] / 2.0, b[0] + b[2] / 2.0);

  float res = (bot - top) * (bot - top) + (right - left) * (right - left);
  return res;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

static float cal_new_iou(vector<float> box, vector<float> truth) {
  float c = box_c(box, truth);

  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  float iou = inter_area * 1.0 / union_area;
  if (c == 0) return iou;

  float d = (box[0] - truth[0]) * (box[0] - truth[0]) +
            (box[1] - truth[1]) * (box[1] - truth[1]);
  float u = pow(d / c, 0.6);
  return iou - u;
}

template <class T>
void print(vector<T> date) {
  for (auto d : date) {
    cout << d << " ";
  }
  cout << endl;
}

void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
              const float nms, const float conf, vector<size_t>& res,
              bool stable, const int iou_type) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  if (stable) {
    stable_sort(
        order.begin(), order.end(),
        [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
          return ls.first > rs.first;
        });
  } else {
    sort(order.begin(), order.end(),
         [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
           return ls.first > rs.first;
         });
  }
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

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
      float ovr = 0.0;
      if (iou_type == 0) {
        ovr = cal_iou(boxes[j], boxes[i]);
      } else {
        ovr = cal_new_iou(boxes[j], boxes[i]);
      }
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}

