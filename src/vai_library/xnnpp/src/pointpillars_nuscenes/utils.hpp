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
#include <map>
#include "vitis/ai/nnpp/pointpillars_nuscenes.hpp"
#include "./anchor.hpp"

namespace vitis {
namespace ai {
namespace pointpillars_nus {

struct ScoreIndex {
  uint32_t index=0;
  int8_t score=0;
};

struct Point {
  float x=0.0;
  float y=0.0;
  Point() {}
  Point(double _x, double _y) { x = _x, y = _y; }

  void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  Point operator+(const Point &b) const {
    return Point(x + b.x, y + b.y);
  }

  Point operator-(const Point &b) const {
    return Point(x - b.x, y - b.y);
  }
};

void sigmoid(float* input, float* output, size_t cls);

std::vector<uint32_t> topK_indexes(const std::vector<float> &scores, uint32_t k);

std::vector<std::vector<float>> 
bbox_decode(const std::vector<uint32_t> &indexes, const Anchors &anchors,  
            const int8_t *bbox_batch_ptr, uint32_t bbox_ndim, float bbox_layer_scale);

std::vector<std::vector<float>> 
bbox_decode_test(const std::vector<uint32_t> &indexes, const Anchors &anchors,  
            const int8_t *bbox_batch_ptr, uint32_t bbox_ndim, float bbox_layer_scale);
std::vector<std::vector<float>> 
get_bboxes_for_nms(const std::vector<std::vector<float>> &bboxes, uint32_t bbox_ndim);
std::vector<std::vector<float>> 
get_bboxes_for_nms_test(const std::vector<std::vector<float>> &bboxes, uint32_t bbox_ndim);


std::vector<std::pair<uint32_t, uint32_t>> 
nms_multiclasses_int8(const int8_t *bbox_layer_ptr, uint32_t bbox_ndim, float bbox_layer_scale,
                      const Anchors &anchors, const vector<vector<ScoreIndex>> &score_indices,
                      std::map<int, std::vector<float>> &bboxes_decoded,
                      uint32_t num_classes, int8_t score_thresh, float nms_thresh, uint32_t max_num);

std::vector<std::pair<uint32_t, uint32_t>> 
nms_3d_multiclasses(const std::vector<std::vector<float>> &bboxes, 
                    const std::vector<std::vector<float>> &scores, 
                    uint32_t num_classes,
                    float score_thresh, float nms_thresh, uint32_t max_num);
 

}}}
