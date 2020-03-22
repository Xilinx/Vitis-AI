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

#include <algorithm>
#include <map>
#include <vector>
#include <iostream>
#include "vitis/ai/nnpp/apply_nms.hpp"

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0)
    return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}
template <class T>
void print(vector <T> date){
  for (auto d : date){
    cout << d << " ";
  }  
  cout << endl;
}

void applyNMS(const vector<vector<float>> &boxes, const vector<float> &scores,
              const float nms, const float conf, vector<size_t> &res) {
  const size_t count = boxes.size();
  multimap<float, size_t, greater<float>> order_map;
  for (size_t i = 0; i < count; ++i) {
    order_map.insert({scores[i], i});
  }
  vector<size_t> ordered;
  transform(order_map.begin(), order_map.end(), back_inserter(ordered), 
            [](auto &km){return km.second;});
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i])
      continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    //cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j])
        continue;
      float ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms)
        exist_box[j] = false;
    }
  }
}

