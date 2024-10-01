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
#include <string.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <array>
#include <vector>
#include "float.h"

namespace vitis {
namespace ai {
namespace x_autonomous3d {

using namespace std;
constexpr float HALF_PI = 3.1415926 / 2;

struct DataInfo {
  std::vector<float> data;
  std::array<int, 3> shape;  // H, W, C
  float scale;
  int tensor_index;

  void resize(int H, int W, int C) {
    shape[0] = H;
    shape[1] = W;
    shape[2] = C;
    data.resize(H * W * C);
  }

  float& at(int h, int w, int c) {
    return data[h * shape[1] * shape[2] + w * shape[2] + c];
  }
};

struct ScoreIndex {
  float score;
  int label;
  int index;

  static bool compare(const ScoreIndex& l, const ScoreIndex& r) {
    return l.score > r.score;
  }
};

constexpr float EPS = 1e-8;

void sigmoid_n(std::vector<float>& src);
void iou_quality_cal(std::vector<float>& src);
void iou_quality_cal(int8_t* output_tensor_ptr, float scale, int size,
                     std::vector<float>& dst);

void heatmap_calculate_with_iou_quality(std::vector<float>& heatmap,
                                        const std::vector<float>& iou_quality,
                                        const std::vector<float>& iou_alpha,
                                        int class_num);
void heatmap_calculate_with_iou_quality(int8_t* hm_tensor_ptr,
                                        float hm_tensor_scale,
                                        const std::vector<float>& iou_quality,
                                        const std::vector<float>& iou_alpha,
                                        int class_num, vector<float>& dst);

int topk(vector<ScoreIndex> input, int k);

void exp_n(std::vector<float>& src);
void atan2_n(std::vector<float>& dst, const std::vector<float>& src);

struct Point {
  float x=0.0;
  float y=0.0;
  Point() {}
  Point(double _x, double _y) { x = _x, y = _y; }

  void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  Point operator+(const Point& b) const { return Point(x + b.x, y + b.y); }

  Point operator-(const Point& b) const { return Point(x - b.x, y - b.y); }
};

void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
              const float nms, const float conf, vector<size_t>& res);

}  // namespace x_autonomous3d
}  // namespace ai
}  // namespace vitis
