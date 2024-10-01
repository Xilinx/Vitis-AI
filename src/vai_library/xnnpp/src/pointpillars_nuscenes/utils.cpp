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
#include <algorithm>
#include <utility>
#include <map>
#include <cstdio>
#include <fstream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;

DEF_ENV_PARAM(DEBUG_NMS, "0")
DEF_ENV_PARAM(DEBUG_NMS_IOU, "0")
DEF_ENV_PARAM(DEBUG_NMS_USE_2D, "0")
DEF_ENV_PARAM(DEBUG_DECODE, "0")
namespace vitis {
namespace ai {
namespace pointpillars_nus {

constexpr float EPS = 1e-8;

void sigmoid(float* input, float* output, size_t cls) {
  for (auto i = 0u; i < cls; i++) {
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }
}

void bbox_decode_kernel(const int8_t *bbox_ptr, uint32_t bbox_ndim, float bbox_layer_scale, 
                        const float *anchor, uint32_t anchor_ndim, float * bbox_decode_ptr) {
  // bbox dim order: x, y, z, w, l, h, r, others
  //                 0, 1, 2, 3, 4, 5, 6, ...
  // a: anchor
  // t: deltas: bbox original output from dpu 
  // g: result

  // za = za + ha / 2;
  auto za = anchor[2] + anchor[5] / 2; 
  // diagonal = sqrt(la * la + wa * wa)
  auto diagonal = std::sqrt(anchor[4] * anchor[4] + anchor[3] * anchor[3]); 
  // xg = xt * diagonal + xa
  bbox_decode_ptr[0] = (*(bbox_ptr + 0)) * bbox_layer_scale * diagonal + anchor[0];  
  // yg = yt * diagonal + ya
  bbox_decode_ptr[1] = (*(bbox_ptr + 1)) * bbox_layer_scale * diagonal + anchor[1];  
  // zg = zt * ha + za
  bbox_decode_ptr[2] = (*(bbox_ptr + 2)) * bbox_layer_scale * anchor[5] + za;

  // lg = exp(lt) * la 
  bbox_decode_ptr[4] = std::exp((*(bbox_ptr + 4)) *bbox_layer_scale) * anchor[4];
  // wg = exp(wt) * wa 
  bbox_decode_ptr[3] = std::exp((*(bbox_ptr + 3)) *bbox_layer_scale) * anchor[3];
  // hg = exp(ht) * ha 
  bbox_decode_ptr[5] = std::exp((*(bbox_ptr + 5)) *bbox_layer_scale) * anchor[5];
  // rg = rt + ra
  bbox_decode_ptr[6] = (*(bbox_ptr + 6)) * bbox_layer_scale + anchor[6]; 
  // zg = zg - hg / 2
  bbox_decode_ptr[2] = bbox_decode_ptr[2] - bbox_decode_ptr[5] / 2;

  for (auto n = 7u; n < bbox_ndim; ++n) {
    // cgs[n] = t + a
    bbox_decode_ptr[n] = (*(bbox_ptr + n)) * bbox_layer_scale + anchor[n];
  }
}

std::vector<std::vector<float>> 
bbox_decode_test(const std::vector<uint32_t> &indexes, const Anchors &anchors,  
            const int8_t *bbox_batch_ptr, uint32_t bbox_ndim, float bbox_layer_scale) {
  auto bbox_max_num = indexes.size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DECODE))
        << "bbox_max_num : " << bbox_max_num
        << "bbox_ndim : " << bbox_ndim
        << "bbox_layer_scale: " << bbox_layer_scale;
  std::vector<std::vector<float>> result(bbox_max_num);
  //for (auto i : indexes) {
  for (auto k = 0u; k < indexes.size(); ++k) {
    auto idx = indexes[k]; 
    result[k].resize(bbox_ndim);
    bbox_decode_kernel(bbox_batch_ptr + idx * bbox_ndim, bbox_ndim, bbox_layer_scale, anchors[idx].data(), anchors[idx].size(), result[k].data()); 
  }
  return result;
}

std::vector<std::vector<float>> 
bbox_decode(const std::vector<uint32_t> &indexes, const Anchors &anchors,  
            const int8_t *bbox_batch_ptr, uint32_t bbox_ndim, float bbox_layer_scale) {
  auto bbox_max_num = indexes.size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DECODE))
        << "bbox_max_num : " << bbox_max_num
        << "bbox_ndim : " << bbox_ndim
        << "bbox_layer_scale: " << bbox_layer_scale;

  std::vector<std::vector<float>> result(bbox_max_num);
  //for (auto i : indexes) {
  for (auto k = 0u; k < indexes.size(); ++k) {
    // bbox dim order: x, y, z, w, l, h, r, others
    //                 0, 1, 2, 3, 4, 5, 6, ...
    // a: anchor
    // t: deltas: bbox original output from dpu 
    // g: result
   
    auto idx = indexes[k]; 
    result[k].resize(bbox_ndim);

    // za = za + ha / 2;
    auto za = anchors[idx][2] + anchors[idx][5] / 2; 
    // diagonal = sqrt(la * la + wa * wa)
    auto diagonal = std::sqrt(anchors[idx][4] * anchors[idx][4] + anchors[idx][3] * anchors[idx][3]); 
    // xg = xt * diagonal + xa
    result[k][0] = (*(bbox_batch_ptr + idx * bbox_ndim + 0)) * bbox_layer_scale * diagonal + anchors[idx][0];  
    // yg = yt * diagonal + ya
    result[k][1] = (*(bbox_batch_ptr + idx * bbox_ndim + 1)) * bbox_layer_scale * diagonal + anchors[idx][1];  
    // zg = zt * ha + za
    result[k][2] = (*(bbox_batch_ptr + idx * bbox_ndim + 2)) * bbox_layer_scale * anchors[idx][5] + za;

    // lg = exp(lt) * la 
    result[k][4] = std::exp((*(bbox_batch_ptr + idx * bbox_ndim + 4)) *bbox_layer_scale) * anchors[idx][4];
    // wg = exp(wt) * wa 
    result[k][3] = std::exp((*(bbox_batch_ptr + idx * bbox_ndim + 3)) *bbox_layer_scale) * anchors[idx][3];
    // hg = exp(ht) * ha 
    result[k][5] = std::exp((*(bbox_batch_ptr + idx * bbox_ndim + 5)) *bbox_layer_scale) * anchors[idx][5];
    // rg = rt + ra
    result[k][6] = (*(bbox_batch_ptr + idx * bbox_ndim + 6)) * bbox_layer_scale + anchors[idx][6]; 
    // zg = zg - hg / 2
    result[k][2] = result[k][2] - result[k][5] / 2;

    for (auto n = 7u; n < bbox_ndim; ++n) {
      // cgs[n] = t + a
      result[k][n] = (*(bbox_batch_ptr + idx * bbox_ndim + n)) * bbox_layer_scale + anchors[idx][n];
    }
  }
  return result;
}

void get_bbox_bev(const float *bbox, uint32_t bbox_ndim,  float *bbox_bev) {
  // bev : bbox[:, [0, 1, 3, 4, 6]]
  constexpr int bev_idx[5] = {0, 1, 3, 4, 6};
  //auto bev_idx= std::vector<int>{0, 1, 3, 4, 6};
  // xywhr => xyxyr
  auto half_w  = bbox[bev_idx[2]] / 2;
  auto half_h  = bbox[bev_idx[3]] / 2;

  bbox_bev[0] = bbox[bev_idx[0]] - half_w;
  bbox_bev[1] = bbox[bev_idx[1]] - half_h;
  bbox_bev[2] = bbox[bev_idx[0]] + half_w;
  bbox_bev[3] = bbox[bev_idx[1]] + half_h;
  bbox_bev[4] = bbox[bev_idx[4]];
  return;
}

std::vector<std::vector<float>> get_bboxes_for_nms_test(const std::vector<std::vector<float>> &bboxes, uint32_t bbox_ndim) {
  auto result = std::vector<std::vector<float>>(bboxes.size());
  for (auto i = 0u; i < bboxes.size(); ++i) {
    result[i].resize(5);
    get_bbox_bev(bboxes[i].data(), bbox_ndim, result[i].data());
  }
  return result;
}


std::vector<std::vector<float>> get_bboxes_for_nms(const std::vector<std::vector<float>> &bboxes, uint32_t bbox_ndim) {
  // bev : bbox[:, [0, 1, 3, 4, 6]]
  auto bev_idx= std::vector<int>{0, 1, 3, 4, 6};
  // xywhr => xyxyr
  auto result = std::vector<std::vector<float>>(bboxes.size());
  for (auto i = 0u; i < bboxes.size(); ++i) {
    result[i].resize(5);
    auto half_w  = bboxes[i][bev_idx[2]] / 2;
    auto half_h  = bboxes[i][bev_idx[3]] / 2;

    result[i][0] = bboxes[i][bev_idx[0]] - half_w;
    result[i][1] = bboxes[i][bev_idx[1]] - half_h;
    result[i][2] = bboxes[i][bev_idx[0]] + half_w;
    result[i][3] = bboxes[i][bev_idx[1]] + half_h;
    result[i][4] = bboxes[i][bev_idx[4]];
  }

  return result;
}

static int check_in_box2d(const float *box, const Point &p) {
  // params: box (5) [x1, y1, x2, y2, angle]
  const float MARGIN = 1e-5;

  float center_x = (box[0] + box[2]) / 2;
  float center_y = (box[1] + box[3]) / 2;
  float angle_cos = cos(-box[4]),
        angle_sin =
            sin(-box[4]);  // rotate the point in the opposite direction of box
  float rot_x =
      (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x;
  float rot_y =
      -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;
  return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
          rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

int check_rect_cross(const Point &p1, const Point &p2,
                                const Point &q1, const Point &q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

static float cross(const Point &a, const Point &b) {
  return a.x * b.y - a.y * b.x;
}

static float cross(const Point &p1, const Point &p2,
                              const Point &p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

int intersection(const Point &p1, const Point &p0,
                                   const Point &q1, const Point &q0,
                                   Point &ans) {
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

static void rotate_around_center(const Point &center, const float angle_cos,
                          const float angle_sin, Point &p) {
  float new_x =
      (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
  float new_y =
      -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p.set(new_x, new_y);
}

static int point_cmp(const Point &a, const Point &b,
                                const Point &center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

static float box_overlap(const float *box_a, const float *box_b) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]
  float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3],
        a_angle = box_a[4];
  float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3],
        b_angle = box_b[4];

  Point center_a((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2);
  Point center_b((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2);

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

static float iou_bev(const float *box_a, const float *box_b) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]
  float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
  float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
  float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

struct ScoreCompare {
  ScoreCompare(const std::vector<std::vector<float>> &scores)
              : scores_(scores) {}
  bool operator() (const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t>&b) {
    return scores_[a.second][a.first] > scores_[b.second][b.first];
  }

  const std::vector<std::vector<float>> &scores_;
}; 


std::vector<uint32_t> 
nms_3d(const std::vector<std::vector<float>> &boxes,
       const std::vector<float> &scores, 
       float score_thresh,
       float nms_overlap_thresh, uint32_t max_num) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)
  static int n = 0; 
  std::vector<uint32_t> result; 
  auto topk = topK_indexes(scores, max_num);
  if (ENV_PARAM(DEBUG_NMS)) {
    auto name = std::string("./nms_score_topk_ori_") + std::to_string(n) + ".txt"; 
    auto o1 = std::ofstream(name);
    for (auto i = 0u; i < scores.size(); ++i) {
      o1 << "top 1000 index:" << i
        << ", score: " << scores[i] << std::endl;
    }
    o1.close();

    name = std::string("./nms_score_topk_") + std::to_string(n) + ".txt"; 
    auto o2 = std::ofstream(name);
    for (auto i = 0u; i < topk.size(); ++i) {
      o2 << "index:" << i << ", top 1000 index:" << topk[i]
        << ", score: " << scores[topk[i]] << std::endl;
    }
    o2.close();
  }

  for (auto i = 0u; i < topk.size(); ++i) {
    if (ENV_PARAM(DEBUG_NMS)) {
      std::cout << "process class: " << n
                << " topk 500: " << i 
                << " topk 1000: " << topk[i]
                << " score: " << scores[topk[i]]
                << " bbox: ";
      for (auto j = 0u; j < 5; ++j) {
        std::cout << boxes[topk[i]][j] << " ";
      }
      std::cout << std::endl;
    }
    if (scores[topk[i]] < score_thresh) {
      continue;
    } 
    bool select = true;
    for (auto j = 0u; j < result.size(); ++j) {
      auto iou = iou_bev(boxes[topk[i]].data(), boxes[result[j]].data());
      if (ENV_PARAM(DEBUG_NMS)) {
        std::cout << "iou of top 1000: " << topk[i] 
                  << " and top 1000: " << result[j]
                  << " iou is : " << iou << std::endl; 
      }
      if (iou >= nms_overlap_thresh) {
        select = false;
        break;
      }
    }
    if (select) {
      if (ENV_PARAM(DEBUG_NMS)) {
        std::cout << "selected topk 1000 : " << topk[i] << std::endl;
      }
      result.emplace_back(topk[i]);
    }
  }

  n++;
  return result;
}

float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}


float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

static void applyNMS_int8(const int8_t *bbox_layer_ptr, uint32_t bbox_ndim, float bbox_layer_scale,
                          const Anchors &anchors, const vector<ScoreIndex>& score_indices,
                          std::map<int, std::vector<float>> &bboxes_decoded,
                          const float nms, const int8_t conf_int8, vector<size_t>& res) {
  const size_t count = score_indices.size();
  //vector<pair<float, size_t>> order;
  //for (size_t i = 0; i < count; ++i) {
  //  order.push_back({scores[i], i});
  //}
  vector<pair<int8_t, size_t>> order; // score, index of score_indice
  for (size_t i = 0; i < count; ++i) {
    order.push_back({score_indices[i].score, i});
  }
  //stable_sort(order.begin(), order.end(),
  //            [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
  //              return ls.first > rs.first;
  //            });
 
  // key: anchor index, value: bbox and bev (size = bbox_ndim + 5)
  //std::map<int, std::vector<float>> bboxes_decoded; 
  
  stable_sort(order.begin(), order.end(),
              [](const pair<int8_t, size_t>& ls, const pair<int8_t, size_t>& rs) {
                return ls.first > rs.first;
              });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  int nms_cnt = 0;
  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i]; // get index of score_indices
    if (!exist_box[i]) continue;
    if (score_indices[i].score < conf_int8) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(i);
    auto anchor_index_i = score_indices[i].index;
    if (bboxes_decoded.count(anchor_index_i) == 0) {
      bboxes_decoded[anchor_index_i] = std::vector<float>(bbox_ndim + 5, 0);
      bbox_decode_kernel(bbox_layer_ptr + anchor_index_i *bbox_ndim, bbox_ndim, bbox_layer_scale, 
                        anchors[anchor_index_i].data(), bbox_ndim, bboxes_decoded[anchor_index_i].data()); 
      get_bbox_bev(bboxes_decoded[anchor_index_i].data(), bbox_ndim, bboxes_decoded[anchor_index_i].data() + bbox_ndim);
    }

    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;

      // decode bbox and get bev
      auto anchor_index_j = score_indices[j].index;
      if (bboxes_decoded.count(anchor_index_j) == 0) {
        bboxes_decoded[anchor_index_j] = std::vector<float>(bbox_ndim + 5, 0);
        bbox_decode_kernel(bbox_layer_ptr + anchor_index_j * bbox_ndim, bbox_ndim, bbox_layer_scale, 
                           anchors[anchor_index_j].data(), bbox_ndim, bboxes_decoded[anchor_index_j].data()); 
        get_bbox_bev(bboxes_decoded[anchor_index_j].data(), bbox_ndim, bboxes_decoded[anchor_index_j].data() + bbox_ndim);
      }
      auto bbox_bev_i = bboxes_decoded[anchor_index_i].data() + bbox_ndim;
      auto bbox_bev_j = bboxes_decoded[anchor_index_j].data() + bbox_ndim;
      float ovr = 0.f;
      if (ENV_PARAM(DEBUG_NMS_USE_2D)) {
        //ovr = cal_iou(boxes[j], boxes[i]);
      } else {
        //ovr = iou_bev(boxes[j].data(), boxes[i].data());
        ovr = iou_bev(bbox_bev_j, bbox_bev_i);
      }
      nms_cnt++;
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
      if (ovr >= nms) exist_box[j] = false;
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_NMS_IOU))
        << "iou cnt: " << nms_cnt;
}


static void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
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
      if (ENV_PARAM(DEBUG_NMS_USE_2D)) {
        //ovr = cal_iou(boxes[j], boxes[i]);
      } else {
        ovr = iou_bev(boxes[j].data(), boxes[i].data());
      }
      nms_cnt++;
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
      if (ovr >= nms) exist_box[j] = false;
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_NMS_IOU))
        << "iou cnt: " << nms_cnt;
}

// return :anchor_index, label
std::vector<std::pair<uint32_t, uint32_t>> 
nms_multiclasses_int8(const int8_t *bbox_layer_ptr, uint32_t bbox_ndim, float bbox_layer_scale,
                      const Anchors &anchors, const vector<vector<ScoreIndex>> &score_indices,
                      std::map<int, std::vector<float>> &bboxes_decoded,
                      uint32_t num_classes, int8_t score_thresh, float nms_thresh, uint32_t max_num) {

  std::vector<std::pair<uint32_t, uint32_t>> result; 
  for (auto i = 0u; i < num_classes; ++i) {
    std::vector<size_t> single_class_result;
    applyNMS_int8(bbox_layer_ptr, bbox_ndim, bbox_layer_scale, 
                  anchors, score_indices[i], bboxes_decoded,
                  nms_thresh, score_thresh, single_class_result); 
    LOG_IF(INFO, ENV_PARAM(DEBUG_NMS)) 
          << "single_class_result[" << i << "] size:" << single_class_result.size();
    if (ENV_PARAM(DEBUG_NMS)) {
      std::cout << "single_class_result[" << i << "] size:" << single_class_result.size() << std::endl;
      for (auto j = 0u; j < single_class_result.size(); ++j) {
        std::cout << "index: " << single_class_result[j]
                  << std::endl;
      }
    }

    for (auto j = 0u; j < single_class_result.size(); ++j) {
      result.emplace_back(std::make_pair(single_class_result[j], i));
      //result.emplace_back(std::make_pair(score_indices[i][single_class_result[j]].index, i));
    }
  }

  std::stable_sort(result.begin(), result.end(),
                   [&](const std::pair<uint32_t, uint32_t> &l, const std::pair<uint32_t, uint32_t> &r){
                    return score_indices[l.second][l.first].score > score_indices[r.second][r.first].score;
                    }); 

  if (result.size() > max_num) {
    result.resize(max_num);
  }
  
  return result;
} 

// return: index, label
std::vector<std::pair<uint32_t, uint32_t>> 
nms_3d_multiclasses(const std::vector<std::vector<float>> &bboxes, 
                    const std::vector<std::vector<float>> &scores, 
                    uint32_t num_classes,
                    float score_thresh, float nms_thresh, uint32_t max_num) {
  std::vector<std::pair<uint32_t, uint32_t>> result; 
  for (auto i = 0u; i < num_classes; ++i) {
    //auto single_class_result = nms_3d(bboxes, scores[i], score_thresh, nms_thresh, max_num);
    std::vector<size_t> single_class_result;
    applyNMS(bboxes, scores[i], nms_thresh, score_thresh, single_class_result); 

    LOG_IF(INFO, ENV_PARAM(DEBUG_NMS)) 
          << "single_class_result[" << i << "] size:" << single_class_result.size();
    if (ENV_PARAM(DEBUG_NMS)) {
      std::cout << "single_class_result[" << i << "] size:" << single_class_result.size() << std::endl;
      for (auto j = 0u; j < single_class_result.size(); ++j) {
        std::cout << "index: " << single_class_result[j]
                  << std::endl;
      }
    }
    for (auto j = 0u; j < single_class_result.size(); ++j) {
      result.emplace_back(std::make_pair(single_class_result[j], i));
    }
  }

  //std::stable_sort(result.begin(), result.end(), 
  //                 [&](std::pair<uint32_t, uint32_t> &a, std::pair<uint32_t, uint32_t> &b)
  //                 {return scores[a.second][a.first] > scores[b.second][b.first];});
  std::stable_sort(result.begin(), result.end(), ScoreCompare(scores)); 

  if (result.size() > max_num) {
    result.resize(max_num);
  }
  
  return result;
} 

std::vector<uint32_t> topK_indexes(const std::vector<float> &scores, uint32_t k) {
  auto min_size = k > scores.size() ? scores.size() : k;
  std::vector<uint32_t> indices(min_size);
  std::vector<std::pair<uint32_t, float>> scores_with_index(scores.size());
  for (auto i = 0u; i < scores.size(); ++i) {
    scores_with_index[i] = std::make_pair(i, scores[i]); 
  }
  auto compare = [&](std::pair<uint32_t, float> a, std::pair<uint32_t, float> b) {return a.second < b.second;};
  std::make_heap(scores_with_index.begin(), scores_with_index.end(), compare);

  for (auto n = 0u; n < min_size; ++n) {
    std::pop_heap(scores_with_index.begin(), scores_with_index.end() - n, compare); 
    indices[n] = (scores_with_index.end() - n - 1)->first;
  }
  return indices;
}

}}}
