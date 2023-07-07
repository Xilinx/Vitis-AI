/*
 * Copyright 2019 xilinx Inc.
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

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <vart/runner_ext.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/tensor/tensor.hpp>

#include "float.h"
using namespace std;
DEF_ENV_PARAM_2(VAI_LIBRARY_MODELS_DIR, ".", std::string)

struct Point {
  float x, y;
  Point() {}
  Point(double _x, double _y) { x = _x, y = _y; }

  void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  Point operator+(const Point& b) const { return Point(x + b.x, y + b.y); }

  Point operator-(const Point& b) const { return Point(x - b.x, y - b.y); }
};

constexpr float EPS = 1e-8;
constexpr int K = 500;
constexpr int out_size_factor = 1;
constexpr float voxel_size[2] = {0.8, 0.8};
constexpr float pc_range[2] = {-51.2, -51.2};
constexpr float post_center_range[6] = {-61.2, -61.2, -12.0, 61.2, 61.2, 12.0};
constexpr int post_max_size = 500;
constexpr float nms_thr = 0.2;
constexpr int bev[5] = {0, 1, 3, 4, 6};
constexpr int HW = 128 * 128;
using namespace std;
static vector<string> find_model_search_path() {
  auto ret = vector<string>{};
  ret.emplace_back(".");
  ret.emplace_back(ENV_PARAM(VAI_LIBRARY_MODELS_DIR));
  ret.emplace_back("/usr/share/vitis_ai_library/models");
  ret.emplace_back("/usr/share/vitis_ai_library/.models");
  return ret;
}

bool filesize(const string& filename) {
  size_t ret = 0u;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = S_ISREG(statbuf.st_mode) ? statbuf.st_size : 0u;
  }
  return ret > 0u;
}

string find_bevdet_1_pt_file(const string& name) {
  if (filesize(name)) {
    return name;
  }
  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/bevdet_1_pt.xmodel";
    if (filesize(ret)) {
      return ret;
    } else {
    }
  }
  LOG(FATAL) << "cannot find [ bevdet_1_pt.xmodel ]";
  return string("");
}

std::pair<std::vector<int8_t>, std::vector<uint32_t>> topK(
    const int8_t scores[], int scores_num, int begin, int step, uint32_t topk,
    bool sorted, float score_threshold) {
  using elem_t = std::pair<int8_t, uint32_t>;
  std::vector<elem_t> queue;
  for (auto j = 0; j < scores_num; j++) {
    if (scores[j * step + begin] > score_threshold) {
      queue.push_back({scores[j * step + begin], j});
    }
  }
  auto min_size = topk > queue.size() ? queue.size() : topk;
  std::stable_sort(queue.begin(), queue.end(),
                   [](const elem_t& x, const elem_t& y) -> bool {
                     return x.first > y.first;
                   });
  std::vector<elem_t> res(queue.begin(), queue.begin() + min_size);
  if (!sorted) {
    std::stable_sort(res.begin(), res.end(),
                     [](const elem_t& x, const elem_t& y) -> bool {
                       return x.second < y.second;
                     });
  }

  std::vector<int8_t> scalars(min_size);
  std::vector<uint32_t> indices(min_size);
  for (auto j = 0u; j < min_size; ++j) {
    scalars[j] = res[j].first;
    indices[j] = res[j].second;
  }
  return std::make_pair(scalars, indices);
}

std::pair<std::vector<int8_t>, std::vector<uint32_t>> topK(
    const int8_t scores[], int scores_num, uint32_t topk, bool sorted,
    float score_threshold) {
  return topK(scores, scores_num, 0, 1, topk, sorted, score_threshold);
}
template <typename T1>
static std::pair<std::vector<T1>, std::vector<uint32_t>> topK(
    const std::vector<T1>& scores, uint32_t topk) {
  using elem_t = std::pair<T1, uint32_t>;
  std::vector<elem_t> queue(scores.size());
  for (auto j = 0u; j < scores.size(); ++j) {
    queue[j].first = scores[j];
    queue[j].second = j;
  }
  auto min_size = topk > queue.size() ? queue.size() : topk;
  std::stable_sort(queue.begin(), queue.end(),
                   [](const elem_t& x, const elem_t& y) -> bool {
                     return x.first > y.first;
                   });

  std::vector<T1> scalars(min_size);
  std::vector<uint32_t> indices(min_size);
  for (auto j = 0u; j < min_size; ++j) {
    scalars[j] = queue[j].first;
    indices[j] = queue[j].second;
  }
  return std::make_pair(scalars, indices);
}
static std::vector<uint32_t> remainder_n(const std::vector<uint32_t>& src,
                                         uint32_t dividend) {
  std::vector<uint32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = src[i] % dividend;
  }
  return dst;
}

static std::vector<uint32_t> division_n(const std::vector<uint32_t>& src,
                                        uint32_t dividend) {
  std::vector<uint32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = src[i] / dividend;
  }
  return dst;
}
template <typename T1>
std::vector<T1> gather_feat(const T1 input[],
                            const std::vector<uint32_t>& index, size_t cat) {
  std::vector<T1> dst(index.size() * cat);
  for (auto i = 0u; i < index.size(); i++) {
    for (size_t j = 0; j < cat; j++) {
      dst[cat * i + j] = input[cat * index[i] + j];
    }
  }
  return dst;
}
template <typename T1>
std::vector<T1> gather_feat(const T1 input[],
                            const std::vector<uint32_t>& index) {
  return gather_feat(input, index, 1);
}
template <typename T1>
std::vector<T1> gather_feat(const std::vector<T1>& input,
                            const std::vector<uint32_t>& index, size_t cat) {
  std::vector<T1> dst(index.size() * cat);
  for (auto i = 0u; i < index.size(); i++) {
    for (size_t j = 0; j < cat; j++) {
      dst[cat * i + j] = input[cat * index[i] + j];
    }
  }
  return dst;
}
template <typename T1>
std::vector<T1> gather_feat(const std::vector<T1>& input,
                            const std::vector<uint32_t>& index) {
  return gather_feat(input, index, 1);
}
inline float sigmoid_n(float src) { return (1. / (1. + exp(-src))); }

static void xywhr2xyxyr(const float box_xywhr[9], float* dst) {
  auto half_w = box_xywhr[bev[2]] / 2;
  auto half_h = box_xywhr[bev[3]] / 2;
  dst[0] = box_xywhr[bev[0]] - half_w;
  dst[1] = box_xywhr[bev[1]] - half_h;
  dst[2] = box_xywhr[bev[0]] + half_w;
  dst[3] = box_xywhr[bev[1]] + half_h;
  dst[4] = box_xywhr[bev[4]];
}

static int check_in_box2d(const float for_box2d[], const float box[],
                          const Point& p) {
  // params: box (5) [x1, y1, x2, y2, angle]
  const float MARGIN = 1e-5;

  auto& center_x = for_box2d[0];
  auto& center_y = for_box2d[1];
  auto& angle_cos = for_box2d[2];
  auto& angle_sin = for_box2d[3];
  // rotate the point in the opposite direction of box
  float rot_x = p.x * angle_cos + p.y * angle_sin + center_x;
  float rot_y = -p.x * angle_sin + p.y * angle_cos + center_y;
  return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
          rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

inline float cross(const Point& a, const Point& b) {
  return a.x * b.y - a.y * b.x;
}

inline float cross(const Point& p1, const Point& p2, const Point& p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}
inline int point_cmp(const Point& a, const Point& b, const Point& center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}
static int check_rect_cross(const Point& p1, const Point& p2, const Point& q1,
                            const Point& q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

static int intersection(const Point& p1, const Point& p0, const Point& q1,
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

inline void calc_first(float box[], Point box_corners[], float for_box2d[]) {
  float x1 = box[0];
  float y1 = box[1];
  float x2 = box[2];
  float y2 = box[3];
  float angle = box[4];
  auto center = Point((x1 + x2) / 2, (y1 + y2) / 2);
  box_corners[0].set(x1, y1);
  box_corners[1].set(x2, y1);
  box_corners[2].set(x2, y2);
  box_corners[3].set(x1, y2);
  float angle_cos = cos(angle), angle_sin = sin(angle);
  // get oriented corners
  for (int k = 0; k < 4; k++) {
    float new_x = (box_corners[k].x - center.x) * angle_cos +
                  (box_corners[k].y - center.y) * angle_sin + center.x;
    float new_y = -(box_corners[k].x - center.x) * angle_sin +
                  (box_corners[k].y - center.y) * angle_cos + center.y;
    box_corners[k].set(new_x, new_y);
  }

  box_corners[4] = box_corners[0];
  for_box2d[2] = cos(-angle);
  for_box2d[3] = sin(-angle);
  for_box2d[0] = -(box[0] + box[2]) / 2. * for_box2d[2] -
                 (box[1] + box[3]) / 2. * for_box2d[3] + (box[0] + box[2]) / 2.;
  for_box2d[1] = (box[0] + box[2]) / 2. * for_box2d[3] -
                 (box[1] + box[3]) / 2. * for_box2d[2] + (box[1] + box[3]) / 2.;
}

static float box_overlap(const std::vector<bool>& is_first, float boxes[][5],
                         size_t a, size_t b, Point box_corners[][5],
                         float for_box2d[][4]) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]

  if (is_first[a]) calc_first(boxes[a], box_corners[a], for_box2d[a]);
  if (is_first[b]) calc_first(boxes[b], box_corners[b], for_box2d[b]);

  auto& box_a_corners = box_corners[a];
  auto& box_b_corners = box_corners[b];

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
    if (check_in_box2d(for_box2d[a], boxes[a], box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(for_box2d[b], boxes[b], box_a_corners[k])) {
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

static float iou_bev(std::vector<bool>& is_first, float box_areas[],
                     Point box_corners[][5], float for_box2d[][4],
                     float boxes[][5], size_t a, size_t b) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]
  if (is_first[a])
    box_areas[a] = (boxes[a][2] - boxes[a][0]) * (boxes[a][3] - boxes[a][1]);
  if (is_first[b])
    box_areas[b] = (boxes[b][2] - boxes[b][0]) * (boxes[b][3] - boxes[b][1]);
  float& sa = box_areas[a];
  float& sb = box_areas[b];
  float s_overlap = box_overlap(is_first, boxes, a, b, box_corners, for_box2d);
  auto res = s_overlap / fmaxf(sa + sb - s_overlap, EPS);
  if (is_first[a]) is_first[a] = false;
  if (is_first[b]) is_first[b] = false;
  return res;
}

cv::Mat resize_and_crop_image(const cv::Mat& image) {
  auto original_height = 900, original_width = 1600;
  auto final_height = 256, final_width = 704;
  auto resize_scale = 0.48;  //

  int resized_width = original_width * resize_scale;
  int resized_height = original_height * resize_scale;
  cv::Size resize_dims(resized_width, resized_height);
  auto crop_h = max(0, resized_height - final_height);
  auto crop_w = int(max(0, (resized_width - final_width) / 2));
  cv::Rect box(crop_w, crop_h, final_width, final_height);

  if (image.size() == resize_dims) {
    return image(box).clone();
  }
  if (image.size() == cv::Size(final_width, final_height)) {
    return image;
  }
  cv::Mat img;
  cv::resize(image, img, resize_dims, 0, 0, cv::INTER_LINEAR);
  auto res = img(box).clone();
  return res;
}

void copy_input_from_image(const std::vector<cv::Mat>& imgs,
                           vart::TensorBuffer* input, std::vector<float> mean,
                           std::vector<float> scale) {
  const xir::Tensor* tensor = input->get_tensor();
  float input_fixed_scale = vart::get_input_scale(tensor);
  vector<float> real_scale{scale[0] * input_fixed_scale,
                           scale[1] * input_fixed_scale,
                           scale[2] * input_fixed_scale};
  // mean[123.675, 116.28 , 103.53 ], 'std'([58.395, 57.12, 57.375] RGB

  auto batch = tensor->get_shape()[0];
  for (auto i = 0; i < (signed)imgs.size() && i < batch; i++) {
    cv::Mat image_resize = resize_and_crop_image(imgs[i]);
    vitis::ai::NormalizeInputDataRGB(image_resize, mean, real_scale,
                                     (int8_t*)input->data({i, 0, 0, 0}).first);
  }
}

void copy_input_from_bin(const std::vector<char>& bin,
                         vart::TensorBuffer* input) {
  size_t batch_size = input->get_tensor()->get_shape()[0];
  CHECK_EQ(input->get_tensor()->get_data_size(), bin.size())
      << input->get_tensor()->get_name();
  auto size_per_batch = input->get_tensor()->get_data_size() / batch_size;
  for (auto i = 0u; i < batch_size; ++i) {
    memcpy((char*)input->data({(int)i, 0, 0, 0}).first,
           bin.data() + i * size_per_batch, size_per_batch);
  }
}

inline int decode(vitis::ai::library::OutputTensor heatmaptb,
                  vitis::ai::library::OutputTensor regtb,
                  vitis::ai::library::OutputTensor heitb,
                  vitis::ai::library::OutputTensor rottb,
                  vitis::ai::library::OutputTensor dimtb,
                  vitis::ai::library::OutputTensor veltb, float score_threshold,
                  uint32_t& first_label, vitis::ai::CenterPointResult* res) {
  auto heatmap_scale = vitis::ai::library::tensor_scale(heatmaptb);
  auto conf_desigmoid = -logf(1.0f / score_threshold - 1.0f) / heatmap_scale;
  int8_t* heatmap = (int8_t*)heatmaptb.get_data(0);
  // process heat
  int cat = heatmaptb.channel;
  auto w = heatmaptb.width;

  std::vector<int8_t> top_scores;
  std::vector<uint32_t> top_inds;
  std::vector<uint32_t> top_clses;
  std::vector<uint32_t> top_ys;
  std::vector<uint32_t> top_xs;
  if (cat == 1) {
    std::tie(top_scores, top_inds) = topK(heatmap, HW, K, true, conf_desigmoid);
    top_clses = std::vector<uint32_t>(top_scores.size(), 0);
    top_ys = division_n(top_inds, w);
    top_xs = remainder_n(top_inds, w);
  } else if (cat == 2) {
    vector<int8_t> topk_scores;
    vector<uint32_t> topk_inds;
    std::tie(topk_scores, topk_inds) =
        topK(heatmap, HW, 0, cat, K, false, conf_desigmoid);
    auto cls_flag = topk_inds.size();
    auto temp = topK(heatmap, HW, 1, cat, K, false, conf_desigmoid);
    topk_scores.insert(topk_scores.end(), temp.first.begin(), temp.first.end());
    topk_inds.insert(topk_inds.end(), temp.second.begin(), temp.second.end());
    std::vector<uint32_t> topk_ind;
    std::tie(top_scores, topk_ind) = topK(topk_scores, K);
    top_inds = gather_feat(topk_inds, topk_ind);
    for (auto&& i : topk_ind) {
      if (i < cls_flag)
        top_clses.push_back(0);
      else
        top_clses.push_back(1);
    }
    top_ys = gather_feat(division_n(topk_inds, w), topk_ind);
    top_xs = gather_feat(remainder_n(topk_inds, w), topk_ind);
  } else {
    LOG(ERROR) << "The number of cat is not supported to be " << cat;
  }
  int res_num = 0;
  if (top_scores.size() > 0) {
    auto reg_scale = vitis::ai::library::tensor_scale(regtb);
    auto hei_scale = vitis::ai::library::tensor_scale(heitb);
    auto rot_scale = vitis::ai::library::tensor_scale(rottb);
    auto dim_scale = vitis::ai::library::tensor_scale(dimtb);
    auto vel_scale = vitis::ai::library::tensor_scale(veltb);
    int8_t* reg = (int8_t*)regtb.get_data(0);
    int8_t* hei = (int8_t*)heitb.get_data(0);
    int8_t* rot = (int8_t*)rottb.get_data(0);
    int8_t* dim = (int8_t*)dimtb.get_data(0);
    int8_t* vel = (int8_t*)veltb.get_data(0);
    // process reg
    auto top_reg = gather_feat(reg, top_inds, 2);
    auto top_hei = gather_feat(hei, top_inds);
    auto top_rot = gather_feat(rot, top_inds, 2);
    auto top_dim = gather_feat(dim, top_inds, 3);
    auto top_vel = gather_feat(vel, top_inds, 2);

    // std::vector<CenterPointResult> res;
    for (auto i = 0u; i < top_scores.size(); i++) {
      if (hei_scale * top_hei[i] >= post_center_range[2] &&
          hei_scale * top_hei[i] <= post_center_range[5]) {
        float xs = (top_xs[i] + reg_scale * top_reg[i * 2]) * out_size_factor *
                       voxel_size[0] +
                   pc_range[0];
        float ys = (top_ys[i] + reg_scale * top_reg[i * 2 + 1]) *
                       out_size_factor * voxel_size[1] +
                   pc_range[1];
        if (xs >= post_center_range[0] && ys >= post_center_range[1] &&
            xs <= post_center_range[3] && ys <= post_center_range[4]) {
          res[res_num].bbox[0] = xs;
          res[res_num].bbox[1] = ys;
          res[res_num].bbox[3] = exp(dim_scale * top_dim[3 * i]);
          res[res_num].bbox[4] = exp(dim_scale * top_dim[3 * i + 1]);
          res[res_num].bbox[5] = exp(dim_scale * top_dim[3 * i + 2]);
          res[res_num].bbox[2] =
              hei_scale * top_hei[i] - res[res_num].bbox[5] * 0.5f;
          res[res_num].bbox[6] =
              atan2(rot_scale * top_rot[2 * i], rot_scale * top_rot[2 * i + 1]);
          res[res_num].bbox[7] = vel_scale * top_vel[2 * i];
          res[res_num].bbox[8] = vel_scale * top_vel[2 * i + 1];
          res[res_num].score = sigmoid_n(heatmap_scale * top_scores[i]);
          res[res_num].label = top_clses[i] + first_label;
          res_num++;
        }
      }
    }
  }
  first_label += cat;
  return res_num;
}

std::vector<vitis::ai::CenterPointResult> post_process(
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    float score_threshold) {
  const std::vector<std::string> heads{"heatmap", "reg", "height",
                                       "rot",     "dim", "vel"};

  vitis::ai::library::OutputTensor out_tensors[6][6];
  for (auto&& o : output_tensors) {
    auto name = o.name;
    size_t j = 0;
    for (j = 0u; j < 6u; ++j) {
      if (name.find(std::string("ModuleList_") + std::to_string(j)) !=
          std::string::npos) {
        break;
      }
    }
    for (size_t i = 0u; i < heads.size(); ++i) {
      if (name.find(heads[i]) != std::string::npos) {
        out_tensors[j][i] = o;
        break;
      }
    }
  }

  std::vector<vitis::ai::CenterPointResult> ress;
  vitis::ai::CenterPointResult decode_res[K];
  float bboxes[K][5];
  float box_areas[K];
  float for_box2d[K][4];
  Point box_corners[K][5];
  uint32_t first_label = 0;
  for (auto&& os : out_tensors) {
    //__TIC__(post_process_decode);
    auto decode_res_num = decode(os[0], os[1], os[2], os[3], os[4], os[5],
                                 score_threshold, first_label, decode_res);
    //__TOC__(post_process_decode);

    //__TIC__(post_process_nms_3d);
    if (decode_res_num > 0) {
      for (auto i = 0; i < decode_res_num; ++i) {
        xywhr2xyxyr(decode_res[i].bbox, bboxes[i]);
      }
      // After decode is sorted
      // Do nms
      vector<bool> exist_box(decode_res_num, true);
      vector<bool> is_first(decode_res_num, true);

      for (int i = 0; i < decode_res_num; ++i) {
        if (!exist_box[i]) continue;
        ress.push_back(decode_res[i]);  // add a box as result
        for (int j = i + 1; j < decode_res_num; ++j) {
          if (!exist_box[j]) continue;
          if (iou_bev(is_first, box_areas, box_corners, for_box2d, bboxes, j,
                      i) > nms_thr)
            exist_box[j] = false;
        }
      }
    }
    //__TOC__(post_process_nms_3d);
  }
  std::vector<float> scores;
  for (auto i = 0u; i < ress.size(); ++i) {
    scores.push_back(ress[i].score);
  }
  return gather_feat(ress, topK(scores, post_max_size).second);
}
int8_t float2fix(float data) {
  int data_max = std::numeric_limits<int8_t>::max();
  int data_min = std::numeric_limits<int8_t>::min();
  auto rlt = 0.0f;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0 && (data - floor(data)) == 0.5) {
    rlt = std::ceil(data);
  } else {
    rlt = std::round(data);
  }
  return rlt;
}

void my_softmax(int8_t* input, int begin_idx, float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < 52; ++i) {
    output[i] = exp(input[i + begin_idx] * 0.5f);
    sum += output[i];
  }
  for (unsigned int i = 0; i < 52; ++i) output[i] = output[i] / sum;
}