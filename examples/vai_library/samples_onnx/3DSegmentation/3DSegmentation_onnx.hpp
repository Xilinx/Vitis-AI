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

#pragma once
#include <assert.h>
#include <glog/logging.h>

#include <opencv2/imgproc/imgproc_c.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <sstream>
#include <vector>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/onnx_task.hpp"
#include "vitis/ai/profiling.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;

using V1F = std::vector<float>;
using V2F = std::vector<V1F>;
using V3F = std::vector<V2F>;
using V1I = std::vector<int>;
using V2I = std::vector<V1I>;

namespace ns_OnnxSegmentation3D {

DEF_ENV_PARAM(ENABLE_KNN, "0")
// const value part
//
#define MAX_POINTS 131072

const float PI = 3.141592653589793;
const float proj_W = 2048.0;
const float proj_H = 64.0;
const float fov = 0.4886921905584123;
const float fov_down = 0.4363323129985824;
std::vector<int> map_inv_{0,  10, 11, 15, 18, 20, 30, 31, 32, 40,
                          44, 48, 49, 50, 51, 70, 71, 72, 80, 81};
const std::vector<float> sensor_means_{12.12, 10.88, 0.23, -1.04, 0.21};
const std::vector<float> sensor_stds_{12.32, 11.47, 6.91, 0.86, 0.16};

int W = 2048;
int center = 12;
float cutoff = 1.0;
float nclasses = 20;
std::vector<float> inv_gauss_k{
    0.9970, 0.9867, 0.9781, 0.9867, 0.9970, 0.9867, 0.9404, 0.9017, 0.9404,
    0.9867, 0.9781, 0.9017, 0.8379, 0.9017, 0.9781, 0.9867, 0.9404, 0.9017,
    0.9404, 0.9867, 0.9970, 0.9867, 0.9781, 0.9867, 0.9970};
int unfold_height = 25;

// return value
struct OnnxSegmentation3DResult {
  int width;
  int height;
  std::vector<int> array;
};

// model class
class OnnxSegmentation3D : public OnnxTask {
 public:
  static std::unique_ptr<OnnxSegmentation3D> create(
      const std::string& model_name) {
    return std::unique_ptr<OnnxSegmentation3D>(
        new OnnxSegmentation3D(model_name));
  }

 protected:
  explicit OnnxSegmentation3D(const std::string& model_name);

  OnnxSegmentation3D(const OnnxSegmentation3D&) = delete;

 public:
  virtual ~OnnxSegmentation3D() { delete[] proj_idx; }
  virtual std::vector<OnnxSegmentation3DResult> run(const V3F& v3f);

 private:
  void preprocess(const V2F& array, int idx);
  void postprocess(OnnxSegmentation3DResult& rs, int idx);

  void post_prec(const std::vector<float>& proj_range,
                 const std::vector<int>& proj_argmax, int idx, V1I&);
  void topk(int idx, float* inv, int k, V1I& out_idx);

 private:
  std::vector<float> input_tensor_values;
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  std::vector<float*> input_tensor_ptr;
  std::vector<float*> output_tensor_ptr;

  bool enable_knn = false;
  V2F depth;
  V2F py, px;
  int* proj_idx;
  V1I pointsize;
  V2F proj_range;
  V1F proj_unfold;
  V1F proj_unfold2;
  V2I idx_list;
  std::shared_ptr<float> k2_distances;
  V1F unproj_unfold_1_argmax;
  V1I knn_idx;
  V1F knn_argmax;
  std::shared_ptr<float> knn_argmax_onehot;
  V1I proj_argmax;
  V1I unproj_argmax;
};

static void norm(const std::vector<float>& input1,
                 const std::vector<float>& input2,
                 const std::vector<float>& input3, std::vector<float>& output) {
  for (size_t i = 0; i < input1.size(); i++) {
    auto x = input1[i];
    auto y = input2[i];
    auto z = input3[i];
    output[i] = sqrt(x * x + y * y + z * z);
  }
}

static void proj_y(const std::vector<float>& z, const std::vector<float>& depth,
                   std::vector<float>& result) {
  static float v1 = proj_H - fov_down * proj_H / fov;
  static float v2 = proj_H / fov;
  size_t size = z.size();
  for (size_t i = 0; i < size; i++) {
    float t = floor(v1 - v2 * (asin(z[i] / depth[i])));
    result[i] = std::clamp(t, 0.0f, 63.0f);
  }
}

static void proj_x(const std::vector<float>& x, const std::vector<float>& y,
                   std::vector<float>& result) {
  static float v1 = 1024.0 / PI;
  size_t size = x.size();
  for (size_t i = 0; i < size; i++) {
    float t = floor(1024.0 - v1 * atan2(y[i], x[i]));
    result[i] = std::clamp(t, 0.0f, 2047.0f);
  }
}

void OnnxSegmentation3D::preprocess(const V2F& array, int idx) {
  pointsize[idx] = array[0].size();
  __TIC__(get_depth)
  norm(array[0], array[1], array[2], depth[idx]);
  __TOC__(get_depth)

  memset(proj_idx, 0, MAX_POINTS * sizeof(int));
  if (enable_knn) {
    proj_range[idx].assign(proj_range[idx].size(), -1);
  }

  __TIC__(get_py)
  proj_y(array[2], depth[idx], py[idx]);
  __TOC__(get_py)
  __TIC__(get_px)
  proj_x(array[0], array[1], px[idx]);
  __TOC__(get_px)

  __TIC__(permute)
  int height = 64;
  int width = 2048;
  float* p = input_tensor_ptr[0] + idx * (5 * MAX_POINTS);  // chw
  int offset[5];
  for (int i = pointsize[idx] - 1; i >= 0; --i) {  // i>=0
    int pyi = py[idx][i], pxi = px[idx][i];

    int flagpos = pyi * 2048 + pxi;
    if (enable_knn) {
      idx_list[idx][i] = flagpos;
    }

    if (proj_idx[flagpos] == 0) {
      proj_idx[flagpos] = 1;

      for (int k = 0; k < 5; k++) {
        offset[k] = k * height * width + pyi * width + pxi;
      }
      p[offset[0]] = (depth[idx][i] - sensor_means_[0]) / sensor_stds_[0];
      p[offset[1]] = (array[0][i] - sensor_means_[1]) / sensor_stds_[1];
      p[offset[2]] = (array[1][i] - sensor_means_[2]) / sensor_stds_[2];
      p[offset[3]] = (array[2][i] - sensor_means_[3]) / sensor_stds_[3];
      p[offset[4]] = (array[3][i] - sensor_means_[4]) / sensor_stds_[4];
      // for (int k=0;k<5;k++) { std::cout <<k<<" "<<pyi<<" "<<pxi<<" X
      // "<<p[offset[k]]<<" X\n"; }
      if (enable_knn) {
        proj_range[idx][flagpos] = depth[idx][i];
      }
    }
  }
  __TOC__(permute)
}

template <class ForwardIt>
ForwardIt max_element_nth(ForwardIt first, ForwardIt last, int step) {
  if (first == last) {
    return last;
  }
  ForwardIt largest = first;
  first += step;
  for (; first < last; first += step) {
    if (*largest < *first) {
      largest = first;
    }
  }
  return largest;
}

void argmax_with_map(float* input, int W, int H, V1I& index) {
  for (int i = 0; i < W; i++) {
    auto start = input + i + 1 * MAX_POINTS;
    index[i] =
        map_inv_[(max_element_nth(start, input + i + (H - 1) * MAX_POINTS,
                                  MAX_POINTS) -
                  start) /
                     MAX_POINTS +
                 1];
  }
}

template <typename T2>
void scatter_add_(float* self, const std::vector<T2>& index, int H, int W) {
  for (auto h = 0; h < H; h++) {
    for (auto w = 0; w < W; w++) {
      auto index_chw = h * MAX_POINTS + w;
      auto self_chw = int(index[index_chw]) * MAX_POINTS + w;
      self[self_chw] += 1;
    }
  }
}

template <typename T1, typename T2>
void gather_1(float* input1, std::vector<T1>& input2, std::vector<T2>& index,
              int index_height, int index_width, std::vector<T1>& output1) {
  for (auto j = 0; j < index_height; j++) {   // 5
    for (auto k = 0; k < index_width; k++) {  // 127405
      auto index_ijk = j * MAX_POINTS + k;
      output1[index_ijk] = input1[index[index_ijk] * MAX_POINTS + k] > cutoff
                               ? nclasses
                               : input2[index[index_ijk] * MAX_POINTS + k];
    }
  }
}

void unfold(const V1F& input1,       // MAX_POINTS
            const V1I& input2,       // MAX_POINTS
            vector<float>& output1,  // 25*MAX_POINTS
            vector<float>& output2)  // 25*MAX_POINTS
{
  const int kernel_height = 5;
  const int kernel_width = 5;
  const int pad_height = 2;
  const int pad_width = 2;
  const int input_height = 64;
  const int input_width = 2048;
  const int output_height = 64;
  const int output_width = 2048;

  for (int64_t c = 0; c < kernel_height * kernel_width; ++c) {  // 5x5
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    for (int64_t h = 0; h < output_height; ++h) {  // 64
      int64_t h_im = h - pad_height + h_offset;
      if (h_im < 0 || h_im >= input_height) continue;
      for (int64_t w = 0; w < output_width; ++w) {  // 2048
        int64_t w_im = w - pad_width + w_offset;
        if (w_im >= 0 && w_im < input_width) {
          output1[(c * output_height + h) * output_width + w] =
              input1[h_im * input_width + w_im];
          output2[(c * output_height + h) * output_width + w] =
              input2[h_im * input_width + w_im];
        }
      }
    }
  }
}

void OnnxSegmentation3D::topk(int idx, float* inv, int k, V1I& out_idx) {
  struct cmp1 {
    bool operator()(const std::pair<int, float>& a,
                    const std::pair<int, float>& b) {
      return std::get<1>(a) <= std::get<1>(b);
    }
  };
  priority_queue<std::pair<int, float>, vector<std::pair<int, float>>, cmp1>
      minHeap;
  float invf = 0.0;
  int pos = 0;
  for (int index = 0; index < pointsize[idx]; ++index) {  // near MAX_POINTS
    for (int i = 0; i < unfold_height; ++i) {             // 25
      invf = inv[index + i * MAX_POINTS];
      if (i < k) {
        minHeap.push(std::make_pair(i, invf));
        continue;
      }
      if (invf >= std::get<1>(minHeap.top())) {
        continue;
      } else {
        minHeap.pop();
        minHeap.push(std::make_pair(i, invf));
      }
    }
    pos = k - 1;
    while (!minHeap.empty()) {
      out_idx[pos * MAX_POINTS + index] = std::get<0>(minHeap.top());
      minHeap.pop();
      pos--;
    }
  }
}

void OnnxSegmentation3D::post_prec(const std::vector<float>& proj_range,
                                   const std::vector<int>& proj_argmax, int idx,
                                   V1I& knn_argmax_out) {
  __TIC__(post_pred_clear)
  proj_unfold.assign(proj_unfold.size(), 0);
  proj_unfold2.assign(proj_unfold2.size(), 0);
  memset(knn_argmax_onehot.get(), 0, 21 * MAX_POINTS);
  __TOC__(post_pred_clear)

  __TIC__(unfold1)
  unfold(proj_range, proj_argmax, proj_unfold, proj_unfold2);  // 25xMAX_POINTS
  __TOC__(unfold1)

  __TIC__(unproj_rang)
  float tmp;
  float* k2_distances_p = k2_distances.get();
  for (auto i = 0; i < unfold_height; i++) {  // 25
    auto offset = MAX_POINTS * i;             //  131072*i
    for (auto j = 0; j < pointsize[idx]; j++) {
      unproj_unfold_1_argmax[offset + j] =
          proj_unfold2[offset + idx_list[idx][j]];
      if (i != 12) {
        if ((tmp = proj_unfold[offset + idx_list[idx][j]]) >= 0) {
          k2_distances_p[offset + j] =
              (abs(tmp - depth[idx][j])) * inv_gauss_k[i];
        } else {
          k2_distances_p[offset + j] = FLT_MAX * inv_gauss_k[i];
        }
      }
    }
  }
  __TOC__(unproj_rang)

  __TIC__(permute_topk)
  topk(idx, k2_distances.get(), 5, knn_idx);
  __TOC__(permute_topk)

  __TIC__(gather)
  gather_1(k2_distances.get(), unproj_unfold_1_argmax, knn_idx, 5,
           pointsize[idx], knn_argmax);
  __TOC__(gather)

  __TIC__(scatter_add)
  scatter_add_(knn_argmax_onehot.get(), knn_argmax, 5, pointsize[idx]);
  __TOC__(scatter_add)

  __TIC__(argmax)
  argmax_with_map(knn_argmax_onehot.get(), (int)pointsize[idx], 21,
                  knn_argmax_out);
  __TOC__(argmax)
}

// segmentation3d postprocess
void OnnxSegmentation3D::postprocess(OnnxSegmentation3DResult& rs, int idx) {
  // in :  1x5xMAX_POINTS
  // out:  1x20xMAX_POINTS

  rs.width = input_shapes_[0][3];
  rs.height = input_shapes_[0][2];
  float* data_int = output_tensor_ptr[0] + idx * (20 * MAX_POINTS);
  int len = output_shapes_[0][2] * output_shapes_[0][3];
  int t_size = output_shapes_[0][1] * len;
  for (int i = 0; i < len; i++) {
    auto max_ind = max_element_nth(data_int + i, data_int + t_size, len);
    auto dis = (max_ind - (data_int + i)) / len;
    proj_argmax[i] = dis;
  }

  unproj_argmax.resize(pointsize[idx]);
  if (!enable_knn) {  // 10ms
    for (int i = 0; i < pointsize[idx]; i++) {
      unproj_argmax[i] = map_inv_[proj_argmax[py[idx][i] * 2048 + px[idx][i]]];
    }
  } else {
    post_prec(proj_range[idx], proj_argmax, idx, unproj_argmax);
  }
  rs.array.swap(unproj_argmax);
}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

OnnxSegmentation3D::OnnxSegmentation3D(const std::string& model_name)
    : OnnxTask(model_name) {
  auto input_shape = input_shapes_[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values_(total_number_elements);
  input_tensor_values_.swap(input_tensor_values);

  auto batch_size = input_shapes_[0][0];

  output_tensor_ptr.resize(1);
  input_tensor_ptr.resize(1);

  enable_knn = ENV_PARAM(ENABLE_KNN);

  proj_range.resize(batch_size);
  depth.resize(batch_size);
  py.resize(batch_size);
  px.resize(batch_size);
  pointsize.resize(batch_size);
  idx_list.resize(batch_size);
  for (int i = 0; i < (int)batch_size; i++) {
    proj_range[i].resize(MAX_POINTS);
    depth[i].resize(MAX_POINTS);
    py[i].resize(MAX_POINTS);
    px[i].resize(MAX_POINTS);
    idx_list[i].resize(MAX_POINTS);
  }
  unproj_argmax.reserve(MAX_POINTS);
  proj_argmax.resize(MAX_POINTS);
  knn_idx.resize(5 * MAX_POINTS);
  knn_argmax.resize(5 * MAX_POINTS);

  proj_idx = new int[MAX_POINTS];
  proj_unfold.resize(25 * MAX_POINTS);
  proj_unfold2.resize(25 * MAX_POINTS);
  unproj_unfold_1_argmax.resize(25 * MAX_POINTS);

  k2_distances = std::shared_ptr<float>(new float[unfold_height * MAX_POINTS]);
  knn_argmax_onehot = std::shared_ptr<float>(new float[21 * MAX_POINTS]);
}

std::vector<OnnxSegmentation3DResult> OnnxSegmentation3D::run(const V3F& v3f) {
  __TIC__(total)
  __TIC__(preprocess)
  if (input_tensors.size()) {
    input_tensors[0] = Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]);
  } else {
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values.data(), input_tensor_values.size(),
        input_shapes_[0]));
  }
  input_tensor_ptr[0] = input_tensors[0].GetTensorMutableData<float>();
  int real_batch = std::min((int)input_shapes_[0][0], (int)v3f.size());
  memset(input_tensor_ptr[0], 0,
         real_batch * 5 * MAX_POINTS * sizeof(float));  //
  for (int i = 0; i < real_batch; i++) {
    preprocess(v3f[i], i);
  }
  __TOC__(preprocess)

  __TIC__(session_run)

  run_task(input_tensors, output_tensors);
  output_tensor_ptr[0] = output_tensors[0].GetTensorMutableData<float>();
  __TOC__(session_run)

  __TIC__(postprocess)
  std::vector<OnnxSegmentation3DResult> ret(real_batch);
  for (int i = 0; i < real_batch; i++) {
    postprocess(ret[i], i);
  }
  __TOC__(postprocess)
  __TOC__(total)
  return ret;
}

}  // namespace ns_OnnxSegmentation3D

