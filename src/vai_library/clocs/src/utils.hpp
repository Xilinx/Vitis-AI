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
#include <vector>
#include <cstdlib>
using std::vector;
namespace vitis {
namespace ai {
namespace clocs {

constexpr float PI = 3.1415926;

using V1F = std::vector<float>;
using V1I = std::vector<int>;
using V2F = std::vector<V1F>;
using V2I = std::vector<V1I>;

struct ScoreIndex {
  float score;
  int label;
  int index;

  static bool compare(const ScoreIndex& l, const ScoreIndex& r) {
    return l.score > r.score;
  }
};

int topk(vector<ScoreIndex> input, int k);

void sigmoid_n(std::vector<float>& src);
void decode_bbox(std::vector<float>& bbox_encode,
                 const std::vector<float>& anchor);

vector<float> transform_for_nms(const vector<vector<float>>& bboxes);

std::vector<float> center_to_corner_box_2d(std::vector<float>& centers,
                                           std::vector<float>& dims,
                                           std::vector<float>& angles);

std::vector<float> corner_to_standup_2d(const std::vector<float>& boxes_corner,
                                        size_t batch, size_t ndim);

std::vector<int> non_max_suppression_cpu(V2F& boxes_in, const V1F& scores_,
                                         int pre_max_size_, int post_max_size_,
                                         float iou_threshold);

}  // namespace clocs
}  // namespace ai
}  // namespace vitis
