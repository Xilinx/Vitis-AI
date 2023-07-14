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

#include "./utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

DEF_ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT, "0");

namespace vitis {
namespace ai {
namespace clocs {

void sigmoid_n(std::vector<float>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = (1. / (1. + exp(-src[i])));
  }
  src.swap(dst);
}

// ndim = 2
std::vector<float> corner_nd_2(std::vector<float>& dims) {
  std::vector<float> corner_norm_2 = {-0.5, -0.5, -0.5, 0.5,
                                      0.5,  0.5,  0.5,  -0.5};
  int ndim = 2;
  int batch = dims.size() / ndim;
  std::vector<float> corners(batch * 8, 0);
  for (auto i = 0; i < batch; ++i) {
    for (auto j = 0; j < 8; ++j) {
      corners[i * 8 + j] = dims[i * 2 + j % 2] * corner_norm_2[j];
    }
  }
  return corners;
}

vector<float> rotation_2d(std::vector<float>& points,
                          const std::vector<float>& angles) {
  // points : [N , 4, 2]
  // angles : [N , 1]
  auto N = angles.size();

  // rot_mat_T : [2, 2, N]
  // rot_mat_T = [[rot_cos, -rot_sin], [rot_sin, rot_cos]]
  vector<float> rot_mat_T(N * 4);
  for (auto i = 0u; i < N; ++i) {
    auto rot_sin = std::sin(angles[i]);
    auto rot_cos = std::cos(angles[i]);
    rot_mat_T[i] = rot_cos;
    rot_mat_T[i + N] = -rot_sin;
    rot_mat_T[i + N * 2] = rot_sin;
    rot_mat_T[i + N * 3] = rot_cos;
  }

  // einsum: 'aij, jka->aik' input: points, rot_mat_T
  vector<float> result(N * 4 * 2);
  for (auto a = 0u; a < N; ++a) {
    for (auto i = 0u; i < 4; ++i) {
      for (auto j = 0u; j < 2; ++j) {
        for (auto k = 0u; k < 2; ++k) {
          //
          result[a * 8 + i * 2 + k] +=
              points[a * 8 + i * 2 + j] * rot_mat_T[j * 2 * N + k * N + a];
        }
      }
    }
  }
  return result;
}

vector<float> center_to_corner_box_2d(vector<float>& centers,
                                      vector<float>& dims,
                                      vector<float>& angles) {
  // centers: [N, 2]
  // dims: [N, 2]
  // angles: [N, 1]
  auto N = angles.size();
  // LOG(INFO) << "N:" << N;
  // corners: [N, 4, 2]
  auto corners = corner_nd_2(dims);
  // LOG(INFO) << "corners:" << N;

  // corners: [N, 4, 2]
  corners = rotation_2d(corners, angles);
  // LOG(INFO) << "corners: size" << corners.size();

  // corners += centers
  for (auto n = 0u; n < N; ++n) {
    for (auto i = 0u; i < 4; ++i) {
      for (auto j = 0u; j < 2; ++j) {
        corners[n * 8 + i * 2 + j] += centers[n * 2 + j];
      }
    }
  }

  // LOG(INFO) << "corners: size" << corners.size();
  return corners;
}

vector<float> corner_to_standup_2d(const vector<float>& boxes_corner,
                                   size_t batch, size_t ndim) {
  // boxes_corner: [batch, K, ndim]
  //
  // transpose : [batch, K, ndim] => [batch, ndim, K]
  vector<float> boxes_transpose(boxes_corner.size());
  auto K = boxes_corner.size() / (batch * ndim);
  for (auto n = 0u; n < batch; ++n) {
    for (auto k = 0u; k < K; ++k) {
      for (auto d = 0u; d < ndim; ++d) {
        boxes_transpose[n * K * ndim + d * K + k] =
            boxes_corner[n * K * ndim + k * ndim + d];
      }
    }
  }
  vector<float> result(batch * 2 * ndim);
  for (auto n = 0u; n < batch; ++n) {
    // i = 0 : min
    // i = 1 : max
    for (auto d = 0u; d < ndim; ++d) {
      result[n * 2 * ndim + d] = *(
          std::min_element(boxes_transpose.data() + n * ndim * K + d * K,
                           boxes_transpose.data() + n * ndim * K + d * K + K));

      result[n * 2 * ndim + ndim + d] = *(
          std::max_element(boxes_transpose.data() + n * ndim * K + d * K,
                           boxes_transpose.data() + n * ndim * K + d * K + K));
    }
  }
  return result;
}
void decode_bbox(std::vector<float>& bbox_encode,
                 const std::vector<float>& anchor) {
  // x, y, z, w, l, h, r
  // 0, 1, 2, 3, 4, 5, 6
  //
  // a: anchor
  // t: bbox

  // za = za + ha / 2
  float za = anchor[2] + anchor[5] / 2;
  // diagonal = sqrt(la^2 + wa^2);
  float diagnoal = std::sqrt(anchor[3] * anchor[3] + anchor[4] * anchor[4]);
  // xg = xt * diagonal + xa
  float xg = bbox_encode[0] * diagnoal + anchor[0];
  // yg = yt * diagonal + ya
  float yg = bbox_encode[1] * diagnoal + anchor[1];
  // zg = zt * ha + za
  float zg = bbox_encode[2] * anchor[5] + za;
  // lg = exp(lt) * la
  float lg = std::exp(bbox_encode[4]) * anchor[4];
  // wg = exp(wt) * wa
  float wg = std::exp(bbox_encode[3]) * anchor[3];
  // hg = exp(ht) * ha
  float hg = std::exp(bbox_encode[5]) * anchor[5];
  // rg = rt + ra
  float rg = bbox_encode[6] + anchor[6];
  // zg = zg - hg / 2
  zg = zg - hg / 2;
  bbox_encode = {xg, yg, zg, wg, lg, hg, rg};
}

vector<float> transform_for_nms(const vector<vector<float>>& bboxes) {
  auto selected_size = bboxes.size();
  auto& selected_bboxes = bboxes;
  vector<float> centers(selected_size * 2);
  vector<float> dims(selected_size * 2);
  vector<float> angles(selected_size);
  for (auto i = 0u; i < selected_size; ++i) {
    centers[i * 2] = selected_bboxes[i][0];
    centers[i * 2 + 1] = selected_bboxes[i][1];
    dims[i * 2] = selected_bboxes[i][3];
    dims[i * 2 + 1] = selected_bboxes[i][4];
    angles[i] = selected_bboxes[i][6];
    if (ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT)) {
      std::cout << "idx:" << i << ", centers:" << centers[i * 2] << ", "
                << centers[i * 2 + 1];
      std::cout << ", dims:" << dims[i * 2] << ", " << dims[i * 2 + 1];
      std::cout << ", angles:" << angles[i] << std::endl;
    }
  }
  auto box_preds_corners = center_to_corner_box_2d(centers, dims, angles);
  if (ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT)) {
    std::cout << "corners:";
    for (auto i = 0u; i < box_preds_corners.size(); ++i) {
      if (i % 8 == 0) {
        std::cout << std::endl;
      }
      std::cout << box_preds_corners[i] << " ";
    }
    std::cout << std::endl;
  }

  auto boxes_for_nms_pre =
      corner_to_standup_2d(box_preds_corners, selected_size, 2);

  if (ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT)) {
    std::cout << "boxes_for_nms_pre size:" << boxes_for_nms_pre.size()
              << std::endl;
    for (auto i = 0u; i < boxes_for_nms_pre.size(); ++i) {
      if (i % 4 == 0) {
        std::cout << std::endl;
      }
      std::cout << boxes_for_nms_pre[i] << " ";
    }
    std::cout << std::endl;
  }

  return boxes_for_nms_pre;
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

std::vector<int> topk(const V1F& scores, int k, V2F& bboxes_in,
                      V2F& bboxes_out) {
  std::vector<int> vout(k);

  struct cmp1 {
    bool operator()(const std::pair<int, float>& a,
                    const std::pair<int, float>& b) {
      return std::get<1>(a) > std::get<1>(b);
    }
  };
  priority_queue<std::pair<int, float>, vector<std::pair<int, float>>, cmp1>
      minHeap;

  for (unsigned int i = 0; i < scores.size(); i++) {
    if (i < (unsigned int)k) {
      minHeap.push(std::make_pair(i, scores[i]));
      continue;
    }
    if (scores[i] <= std::get<1>(minHeap.top())) {
      continue;
    }
    if (scores[i] > std::get<1>(minHeap.top())) {
      minHeap.pop();
      minHeap.push(std::make_pair(i, scores[i]));
    }
  }
  int pos = k - 1;
  while (!minHeap.empty()) {
    vout[pos] = std::get<0>(minHeap.top());
    bboxes_out[pos].swap(bboxes_in[vout[pos]]);
    minHeap.pop();
    pos--;
  }
  return vout;
}

std::vector<int> non_max_suppression_cpu(V2F& boxes_in, const V1F& scores_,
                                         int pre_max_size_, int post_max_size_,
                                         float iou_threshold) {
  float eps = 1.0;
  int bsize = std::min((int)scores_.size(), pre_max_size_);

  V2F boxes(bsize);
  V1I indices = topk(scores_, bsize, boxes_in, boxes);

  std::vector<int> keep;
  keep.reserve(post_max_size_);
  std::vector<bool> suppressed_rw(bsize, false);
  std::vector<float> area_rw(bsize, 0.0);

  int i_idx, j_idx;
  float xx1, xx2, w, h, inter, ovr;
  for (int i = 0; i < bsize; ++i) {
    area_rw[i] =
        (boxes[i][2] - boxes[i][0] + eps) * (boxes[i][3] - boxes[i][1] + eps);
  }
  for (int i = 0; i < bsize; i++) {
    i_idx = i;
    if (suppressed_rw[i_idx] == true) {
      continue;
    }
    if ((int)keep.size() < post_max_size_) {
      keep.emplace_back(indices[i_idx]);
    } else {
      return keep;
    }
    for (int j = i + 1; j < bsize; j++) {
      j_idx = j;
      if (suppressed_rw[j_idx] == true) {
        continue;
      }
      xx2 = std::min(boxes[i_idx][2], boxes[j_idx][2]);
      xx1 = std::max(boxes[i_idx][0], boxes[j_idx][0]);
      w = xx2 - xx1 + eps;
      if (w > 0) {
        xx2 = std::min(boxes[i_idx][3], boxes[j_idx][3]);
        xx1 = std::max(boxes[i_idx][1], boxes[j_idx][1]);
        h = xx2 - xx1 + eps;
        if (h > 0) {
          inter = w * h;
          ovr = inter / (area_rw[i_idx] + area_rw[j_idx] - inter);
          if (ovr >= iou_threshold) {
            suppressed_rw[j_idx] = true;
          }
        }  // end of if(h>0)
      }    // end of if(w>0)
    }      // end of for(j
  }        // end of for(i
           // std:cout << "keep.size:"<<keep.size() << "   " << bsize << "\n";
  return keep;
}

}  // namespace clocs
}  // namespace ai
}  // namespace vitis
