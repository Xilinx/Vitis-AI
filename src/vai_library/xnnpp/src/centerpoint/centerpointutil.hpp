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
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <string.h>
#include "float.h"
#include <sstream>
#include <algorithm>
#include <string>
#include <map>
#include <tuple>
using namespace std;

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

struct ScoreCompare {
  ScoreCompare(const std::vector<std::vector<float>> &scores)
              : scores_(scores) {}
  bool operator() (const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t>&b) {
    return scores_[a.second][a.first] > scores_[b.second][b.first];
  }

  const std::vector<std::vector<float>> &scores_;
}; 


constexpr float EPS = 1e-8;

template <class T> 
static void readfile(string& filename, vector<T>& data) {
  ifstream input_file(filename);
  std::string line;
  while (std::getline(input_file, line)) {
    istringstream ss(line);
    T num;
    ss >> num;
    data.push_back(num);
  }
  //cout << filename << " " << data.size() << endl;
}


using namespace std;
template <class T> 
struct T3D {
  std::vector<T> data;
  size_t height=0;
  size_t width=0;
  size_t channel=0;
  bool check_dim() {
    if (height * width * channel == data.size()) {
      return true;
    } else {
      std::cout << "Size: " <<  data.size() << "\nC/H/W " << channel << "/" << height << "/" << width << std::endl;
      return false;
    }
  }
  void print() {
    cout << "C/H/W " << channel << "/" << height << "/" << width << endl;
  }
  void resize(size_t c, size_t h, size_t w) {
    data.resize(c * h * w);
    reshape(c, h, w);
  }
  void reshape(size_t c, size_t h, size_t w) {
    height = h;
    width = w;
    channel = c;
    auto check = check_dim();
    if(check == false)
      std::cout << "Shape Error" << std::endl;
  }
  void expand_2(size_t dim) {
    std::vector<T> dst;
    dst.reserve(data.size() * dim);
    for (auto i = 0u; i < data.size(); i++) {
      for (auto j = 0u; j < dim; j++) {
        dst.emplace_back(data[i]);
      }
    }
    data.swap(dst);
    reshape(channel, height, width * dim);
  }
  T at(size_t c, size_t h, size_t w) {
    return data[c*width*height + h*width + w];
  }
};


std::pair<std::vector<float>, std::vector<uint32_t>> topK(const std::vector<float> &scores, uint32_t k) {
  auto min_size = k > scores.size() ? scores.size() : k;
  std::vector<uint32_t> indices(min_size);
  std::vector<float> scalars(min_size);
  std::vector<std::pair<uint32_t, float>> scores_with_index(scores.size());
  for (auto i = 0u; i < scores.size(); ++i) {
    scores_with_index[i] = std::make_pair(i, scores[i]); 
  }
  auto compare = [&](std::pair<uint32_t, float> a, std::pair<uint32_t, float> b) {return a.second < b.second;};
  std::make_heap(scores_with_index.begin(), scores_with_index.end(), compare);

  for (auto n = 0u; n < min_size; ++n) {
    std::pop_heap(scores_with_index.begin(), scores_with_index.end() - n, compare); 
    indices[n] = (scores_with_index.end() - n - 1)->first;
    scalars[n] = (scores_with_index.end() - n - 1)->second;
  }
  return std::make_pair(scalars,indices);
}


void remainder_n(std::vector<uint32_t>& src, uint32_t dividend) {
  std::vector<uint32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = src[i] % dividend;
  }
  src.swap(dst);
}

void division_n(std::vector<uint32_t>& src, uint32_t dividend) {
  std::vector<uint32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = (int)((float)src[i] / (float)dividend);
  }
  src.swap(dst);
}
template <typename T>
void division_n(const std::vector<uint32_t>& src, std::vector<T>& dst, float dividend) {
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = (T)src[i] / (T)dividend;
  }
}

template <typename T1, typename T2>
T3D<T1> gather_1(T3D<T1>& input, T3D<T2>& index) {
  T3D<T1> dst;
  dst.data.resize(index.data.size());
  for (auto i = 0u; i < index.channel; i++) {
    for (auto j = 0u; j < index.height; j++) {
      for (auto k = 0u; k < index.width; k++) {
        auto index_ijk = i*index.height*index.width + j*index.width + k;
        dst.data[index_ijk] = input.data[i*index.height*index.width + index.data[index_ijk]*index.width + k];
      }
    }
  }
  dst.reshape(index.channel, index.height, index.width);
  return dst;
}

template <typename T>
void permute(T3D<T>& input) {
  std::vector<T> dst(input.data.size());
  auto c = input.channel;
  auto h = input.height;
  auto w = input.width;
  for (auto i = 0u; i < c; i++) {
    for (auto j = 0u; j < h; j++) {
      for (auto k = 0u; k < w;k++) {
        dst[j*w*c +k*c +i] = input.data[i*h*w + j*w + k];
      }
    }
  }
  input.data.swap(dst);
  input.reshape(h, w, c);
}


template <typename T1, typename T2>
void gather_feat(T3D<T1>& feats, T3D<T2> inds) {
  inds.expand_2(feats.width);
  /*
  cout << "print feat" << endl;
  for (auto a = ; a < 300; a++) 
    std::cout << feats.data[a]  << " ";
  std::cout << std::endl;
  */
  
  auto feats1 = gather_1(feats, inds);
  feats.data.swap(feats1.data);
  feats.channel = feats1.channel;
  feats.width = feats1.width;
  feats.height = feats1.height;
}

template <typename T1, typename T2>
void transpose_and_gather_feat(T3D<T1>& feat, T3D<T2> ind) {
  permute(feat);
  feat.reshape(1, feat.data.size() / feat.width, feat.width);
  gather_feat(feat, ind);
}

// topk_score, topk_inds, topk_clses, topk_ys, topk_xs
std::tuple<T3D<float>, T3D<uint32_t>, T3D<uint32_t>, T3D<uint32_t>, T3D<uint32_t>>
topK(T3D<float> scores, uint32_t K) {
  auto cat = scores.channel;
  auto height = scores.height;
  auto width = scores.width;
  auto task_len = scores.data.size() / cat;
  //std::cout <<" cat " << cat << " " << task_len << std::endl;
  T3D<float> topk_scores; 
  T3D<uint32_t> topk_inds; 
  T3D<uint32_t> topk_ys; 
  T3D<uint32_t> topk_xs; 
  for (size_t c = 0; c < cat; c++) {
    auto start = scores.data.begin() + c * task_len;
    auto end = scores.data.begin() + c * task_len + task_len;
    std::vector<float>  per_channel(start, end);
    auto temp = topK(per_channel, K); 
    // for (auto p = 0u; p < 100; p++)
      //cout << "topk " << temp.second[p] << " " << temp.first[p] << endl;
    /*
    std::string n1 = "topk_inds.txt";
    std::string n2 = "topk_scores.txt";
    temp.first.resize(0);
    temp.second.resize(0);
    readfile(n1, temp.second);
    readfile(n2, temp.first);
    */
    // for (auto p = 0u; p < 500; p++)
      //cout << "topk " << temp.first[p] << " " << temp.second[p] << endl;
    remainder_n(temp.second, height * width); //% for top_k_inds
    topk_scores.data.insert(topk_scores.data.end(), temp.first.begin(), temp.first.end()); // save topk score
    topk_inds.data.insert(topk_inds.data.end(), temp.second.begin(), temp.second.end()); //save topk ind
    auto temp_ys = temp.second; //init topk ys
    auto temp_xs = temp.second; //init topk xs
    division_n(temp_ys, width); // get ys 
    remainder_n(temp_xs, width);// get xs
    topk_ys.data.insert(topk_ys.data.end(), temp_ys.begin(), temp_ys.end()); // topk_ys
    topk_xs.data.insert(topk_xs.data.end(), temp_xs.begin(), temp_xs.end()); // topk_xs
  }
  /*
  cout << "topk inds " << endl;
  for (auto p = 0u; p < 100; p++) {
    cout << topk_inds.data[p] << " ";
  }
  cout << endl;
  */
  topk_inds.reshape(1, topk_inds.data.size(), 1); // reshape topk_inds.view(batch, -1, 1)
  topk_ys.reshape(1, topk_ys.data.size(), 1); // reshape topk_inds.view(batch, -1, 1)
  topk_xs.reshape(1, topk_xs.data.size(), 1); // reshape topk_inds.view(batch, -1, 1)
  auto topk = topK(topk_scores.data, K);
  /*
  std::string n3 = "topk_ind.txt";
  std::string n4 = "topk_score.txt";
  topk.first.resize(0);
  topk.second.resize(0);
  readfile(n3, topk.second);
  readfile(n4, topk.first);
  */
  T3D<uint32_t> topk_ind; 
  T3D<float> topk_score; 
  topk_ind.data.swap(topk.second);
  topk_ind.reshape(1, topk_ind.data.size(), 1); // topk_inds.view(batch, -1, 1)
  topk_score.data.swap(topk.first);
  topk_score.reshape(1, topk_score.data.size(), 1); // topk_inds.view(batch, -1, 1)
  
  auto topk_clses = topk_ind; // topk_ind / torch.tensor(K, dtype=torch.float)).int()
  division_n(topk_clses.data, K);
  //std::cout << std::endl;
  gather_feat(topk_inds, topk_ind); //gather
  //exit(0);
  gather_feat(topk_ys, topk_ind); //gather
  gather_feat(topk_xs, topk_ind); //gather

 /* 
    for (auto a = 0u; a < 5; a++) 
      std::cout << topk_ind.data[a]  << "/" <<  topk_score.data[a] << " ";
    std::cout << std::endl;
    for (auto a = 0u; a < 5; a++) 
      std::cout << clses[a]  << " ";
    std::cout << std::endl;
    */
  return std::make_tuple(topk_score, topk_inds, topk_clses, topk_ys, topk_xs); 
}


static void sigmoid_n(std::vector<float>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = (1. / (1. + exp(-src[i])));
  }
  src.swap(dst);
}

static void exp_n(std::vector<float>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = exp(src[i]);
  }
  src.swap(dst);
}

T3D<float> atan2_(T3D<float> x, T3D<float> y) {
  T3D<float> dst;
  dst.resize(x.channel, x.height, x.width);
  for(auto i = 0u; i < x.data.size(); i++) {
    dst.data[i] = atan2(x.data[i], y.data[i]);
  }
  return dst;
}

std::vector<std::vector<float>> xywhr2xyxyr(const std::vector<std::vector<float>>& boxes_xywhr, const std::vector<int>& bev) {
  std::vector<std::vector<float>> boxes(boxes_xywhr.size());
  for(auto i = 0u; i < boxes_xywhr.size(); i++) {
    boxes[i].resize(bev.size());
    auto half_w  = boxes_xywhr[i][bev[2]] / 2;
    auto half_h  = boxes_xywhr[i][bev[3]] / 2;
    boxes[i][0] = boxes_xywhr[i][bev[0]] - half_w;
    boxes[i][1] = boxes_xywhr[i][bev[1]] - half_h;
    boxes[i][2] = boxes_xywhr[i][bev[0]] + half_w;
    boxes[i][3] = boxes_xywhr[i][bev[1]] + half_h;
    boxes[i][4] = boxes_xywhr[i][bev[4]];
  }
  return boxes;
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


static float cross(const Point &a, const Point &b) {
    return a.x * b.y - a.y * b.x;
}


static float cross(const Point &p1, const Point &p2,
                              const Point &p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}


int check_rect_cross(const Point &p1, const Point &p2,
                                const Point &q1, const Point &q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
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
/*
    LOG_IF(INFO, ENV_PARAM(DEBUG_NMS)) 
          << "single_class_result[" << i << "] size:" << single_class_result.size();
    if (ENV_PARAM(DEBUG_NMS)) {
      std::cout << "single_class_result[" << i << "] size:" << single_class_result.size() << std::endl;
      for (auto j = 0u; j < single_class_result.size(); ++j) {
        std::cout << "index: " << single_class_result[j]
                  << std::endl;
      }
    }
*/
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

