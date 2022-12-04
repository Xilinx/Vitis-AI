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
#include "./3Dsegmentation_imp.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
#include "float.h"
#include "vitis/ai/env_config.hpp"
#include <fstream>
#include <numeric>
#include <queue>

using namespace std;

namespace vitis {
namespace ai {

#define MAX_POINTS 131072

const float PI = 3.141592653589793;
const float proj_W = 2048.0;
const float proj_H = 64.0;
const float fov = 0.4886921905584123;
const float fov_down = 0.4363323129985824;
std::vector<int>  map_inv_{0, 10, 11, 15, 18,
                         20, 30, 31, 32, 40,
                         44, 48, 49, 50, 51,
                         70, 71, 72, 80, 81};
const std::vector<float> sensor_means_{12.12, 10.88, 0.23, -1.04, 0.21};
const std::vector<float> sensor_stds_{12.32, 11.47, 6.91, 0.86, 0.16};

int W = 2048;
int center = 12;
float cutoff = 1.0;
float nclasses = 20;
std::vector<float> inv_gauss_k{0.9970, 0.9867, 0.9781, 0.9867, 0.9970, 0.9867, 0.9404
                    , 0.9017, 0.9404, 0.9867, 0.9781, 0.9017, 0.8379, 0.9017
                    , 0.9781, 0.9867, 0.9404, 0.9017, 0.9404, 0.9867, 0.9970
                    , 0.9867, 0.9781, 0.9867, 0.9970};
int unfold_height = 25;
int unfold_width = 131072;

static void norm(const std::vector<float>& input1, const std::vector<float>& input2, const std::vector<float>& input3,  std::vector<float>& output) {
  for(size_t i = 0; i < input1.size(); i++) {
    auto x = input1[i];
    auto y = input2[i];
    auto z = input3[i];
    output[i] = sqrt(x*x + y*y +z*z);
  }
}

static void proj_y(const std::vector<float>& z, const std::vector<float>& depth, std::vector<float>& result) {
  static float v1 = proj_H - fov_down*proj_H/fov;
  static float v2 = proj_H/fov;
  size_t size = z.size();
  for (size_t i = 0; i < size; i++) {
    float t = floor(v1 - v2*(asin(z[i] / depth[i])));
    result[i] = std::clamp(t, 0.0f,  63.0f);
  }
}

static void proj_x(const std::vector<float>& x, const std::vector<float>& y, std::vector<float>& result) {
  static float v1 = 1024.0/PI;
  size_t size = x.size();
  for(size_t i = 0; i < size; i++) {
    float t = floor(1024.0 - v1* atan2(y[i], x[i]));
    result[i] = std::clamp(t, 0.0f,  2047.0f);
  }
}

Segmentation3DImp::Segmentation3DImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Segmentation3D>(model_name, need_preprocess) {

  enable_knn = configurable_dpu_task_->getConfig().segmentation_3d_param().enable_knn();
  in_scale = vitis::ai::library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);
  auto batch_size = get_input_batch();

  proj_range.resize(batch_size);
  input_ptr.resize(batch_size);
  output_ptr.resize(batch_size);
  depth.resize(batch_size);
  py.resize(batch_size);
  px.resize(batch_size);
  pointsize.resize(batch_size);
  idx_list.resize(batch_size);
  for(int i=0; i<(int)batch_size; i++) {
    proj_range[i].resize(64*2048);
    depth[i].resize(MAX_POINTS);
    py[i].resize(MAX_POINTS);
    px[i].resize(MAX_POINTS);
    idx_list[i].resize(MAX_POINTS);
    input_ptr[i]  = (int8_t*)configurable_dpu_task_->getInputTensor( )[0][0].get_data(i);
    output_ptr[i] = (int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(i);
  }
  unproj_argmax.reserve(MAX_POINTS);
  proj_argmax.resize(MAX_POINTS);
  knn_idx.resize(5*MAX_POINTS);
  knn_argmax.resize(5*MAX_POINTS);

  proj_idx = new int[64*2048];

  sensor_std_scale.resize(sensor_means_.size());
  sensor_mean_std_scale.resize(sensor_means_.size());
  for(int i=0; i<(int)sensor_means_.size(); ++i) {
    sensor_std_scale[i] = (float)in_scale/sensor_stds_[i];
    sensor_mean_std_scale[i] = (sensor_means_[i]*in_scale)/sensor_stds_[i];
  }
  size_all = int(proj_W * proj_H * 20);
  proj_unfold.resize(25*64*2048);
  proj_unfold2.resize(25*64*2048);
  unproj_unfold_1_argmax.resize(25*64*2048);

  k2_distances = std::shared_ptr<float>(new float[unfold_height*unfold_width]);
  knn_argmax_onehot = std::shared_ptr<float>(new float [21*MAX_POINTS]);
}

Segmentation3DImp::~Segmentation3DImp() {  delete []proj_idx; }

void Segmentation3DImp::preprocess(const V2F& array, int idx) {
  pointsize[idx] = array[0].size();

  __TIC__(get_depth)
  norm(array[0], array[1], array[2], depth[idx]);
  __TOC__(get_depth)

  __TIC__(assign)
  memset (proj_idx, 0, 64*2048*sizeof(int));
  memset(input_ptr[idx], 0, 5*64*2048);
  if (enable_knn) {
    proj_range[idx].assign( proj_range[idx].size(), -1);
  }
  __TOC__(assign)

  __TIC__(get_py)
  proj_y(array[2], depth[idx], py[idx]);
  __TOC__(get_py)
  __TIC__(get_px)
  proj_x(array[0], array[1], px[idx]);
  __TOC__(get_px)

  __TIC__(permute_pre)
  int channels = 5;
  int width = 2048;
  int8_t* p = input_ptr[idx];
  int offset= 0;

  for(int i = pointsize[idx]-1; i >= 0 ; --i) {  // i>=0
    int pyi = py[idx][i], pxi = px[idx][i];
    int flagpos = pyi*2048+pxi;
    if (enable_knn) {
       idx_list[idx][i] = flagpos;
    } 

    if( proj_idx[flagpos] == 0 ) {
       proj_idx[flagpos] = 1;
       offset = pyi*width*channels + pxi*channels;
       p[offset + 0] = (int8_t)std::clamp(int(std::round( depth[idx][i] * sensor_std_scale[0] - sensor_mean_std_scale[0])), -128, 127);
       p[offset + 1] = (int8_t)std::clamp(int(std::round( array[0][i]   * sensor_std_scale[1] - sensor_mean_std_scale[1])), -128, 127);
       p[offset + 2] = (int8_t)std::clamp(int(std::round( array[1][i]   * sensor_std_scale[2] - sensor_mean_std_scale[2])), -128, 127);
       p[offset + 3] = (int8_t)std::clamp(int(std::round( array[2][i]   * sensor_std_scale[3] - sensor_mean_std_scale[3])), -128, 127);
       p[offset + 4] = (int8_t)std::clamp(int(std::round( array[3][i]   * sensor_std_scale[4] - sensor_mean_std_scale[4])), -128, 127);

       if (enable_knn) {
          proj_range[idx][flagpos] = depth[idx][i];
       }
    }
  }
  __TOC__(permute_pre)
}

void  Segmentation3DImp::postprocess(Segmentation3DResult& rs, int idx) {
  int8_t max_v;
  int max_pos;
  int8_t* p = output_ptr[idx];

  for (int i = 0, j=0; i < size_all; i = i + 20, j++) {  // 37ms
    max_v = -128;
    max_pos = 0;
    for(int k=0;k<20;k++) {
      if( p[k] > max_v ) {
         max_v = p[k];  max_pos = k;
      }
    }
    // max_element_nth(p,p+20,1)-p;
    proj_argmax[j] = max_pos;
    p += 20;
  }

  unproj_argmax.resize(pointsize[idx]);
  if(!enable_knn) { 
    for(int i = 0; i < pointsize[idx]; i++) {
      unproj_argmax[i] = map_inv_[ proj_argmax[py[idx][i]*2048 + px[idx][i]] ];
    }
  } else {
    post_prec(proj_range[idx], proj_argmax, idx, unproj_argmax);
  }
  rs.array.swap(unproj_argmax);
}

Segmentation3DResult Segmentation3DImp::run(V2F& array) {
  __TIC__(SEGMENTATION3D)
  __TIC__(pre)
  preprocess(array, 0);
  __TOC__(pre)
  __TIC__(DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPU)
  __TIC__(post)
  Segmentation3DResult rs{getInputWidth(), getInputHeight()};
  postprocess( rs , 0);
  __TOC__(post)
  __TOC__(SEGMENTATION3D)
  return rs;
}

std::vector<Segmentation3DResult> Segmentation3DImp::run(std::vector<std::vector<std::vector<float>>>& arrays) {
  __TIC__(SEGMENTATION3D)
  __TIC__(pre)
  int real_batch = std::min(get_input_batch(), arrays.size());
  for(int i=0; i<real_batch; i++) {
    preprocess(arrays[i], i);
  }
  __TOC__(pre)
  __TIC__(DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPU)
  __TIC__(post)

  std::vector<Segmentation3DResult> rs(real_batch);
  for(int i=0; i<real_batch; i++) {
    rs[i].width  = getInputWidth();
    rs[i].height = getInputHeight();
    postprocess( rs[i] , i);
  }
  __TOC__(post)
  __TOC__(SEGMENTATION3D)
  return rs;
}

template<class ForwardIt>
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
  for(int i=0;i<W;i++) {
    auto start = input+i+1*unfold_width;
    index[i] = map_inv_[(max_element_nth(start, input+i+(H-1)*unfold_width, unfold_width) - start)/unfold_width+1];
  }
}

template<typename T2>
void scatter_add_(float* self, const std::vector<T2>& index, int H, int W ) {
  for (auto h = 0; h < H; h++) {
     for (auto w = 0; w < W; w++) {
        auto index_chw = h*unfold_width + w;
        auto self_chw = int(index[index_chw])*unfold_width + w;
        self[self_chw] += 1;
     }
  }
}

template <typename T1, typename T2>
void gather_1(float* input1, std::vector<T1>& input2, std::vector<T2>& index, int index_height, int index_width, std::vector<T1>& output1 ) {
  for (auto j = 0; j < index_height; j++) {  // 5
    for (auto k = 0; k < index_width; k++) {  // 127405
      auto index_ijk = j*unfold_width+ k;
      output1[index_ijk] = input1[ index[index_ijk]*unfold_width + k] > cutoff ? nclasses : input2[ index[index_ijk]*unfold_width + k];
    }
  }
}

void unfold(
  const V1F& input1,  // 64*2048
  const V1I& input2,  // 64*2048
  vector<float>& output1,  // 25*64*2048 
  vector<float>& output2 )  // 25*64*2048 
{
  const int kernel_height = 5;
  const int kernel_width = 5;
  const int pad_height = 2;
  const int pad_width = 2;
  const int input_height = 64;
  const int input_width = 2048;
  const int output_height = 64;
  const int output_width = 2048;

  for (int64_t c = 0; c <  kernel_height * kernel_width; ++c) {    // 5x5
      int64_t w_offset = c % kernel_width;
      int64_t h_offset = (c / kernel_width) % kernel_height;
      for (int64_t h = 0; h < output_height; ++h) { // 64
          int64_t h_im = h - pad_height + h_offset;
          if (h_im<0|| h_im>=input_height ) continue;
          for (int64_t w = 0; w < output_width; ++w) { // 2048
              int64_t w_im = w - pad_width + w_offset;
              if ( w_im >= 0 && w_im < input_width) {
                output1[(c * output_height + h) * output_width + w] = input1[ h_im * input_width + w_im];
                output2[(c * output_height + h) * output_width + w] = input2[ h_im * input_width + w_im];
              }
          }
      }
  }
}

void Segmentation3DImp::topk(int idx, float* inv, int k, V1I& out_idx )
{  
  struct cmp1 {
    bool operator ()(const std::pair<int, float>& a, const std::pair<int, float>& b ) {
      return std::get<1>(a) <= std::get<1>(b);
    }
  };
  priority_queue<std::pair<int,float>, vector<std::pair<int, float>>,cmp1> minHeap;
  float invf = 0.0;
  int pos=0;
  for(int index=0; index<pointsize[idx]; ++index){  // near 64*2048
    for( int i=0; i<unfold_height; ++i) {  // 25
      invf = inv[ index + i*unfold_width];
      if (i<k) {
        minHeap.push(std::make_pair(i, invf) );
        continue;
      }
      if (invf>= std::get<1>(minHeap.top())) {
        continue;
      } else {
        minHeap.pop();
        minHeap.push(std::make_pair(i, invf) );
      }
    }
    pos = k-1;
    while(!minHeap.empty()){
      out_idx[pos*unfold_width+index] = std::get<0>(minHeap.top());
      minHeap.pop();
      pos--;
    }
  }   
}

void Segmentation3DImp::post_prec(const std::vector<float>& proj_range, 
                           const std::vector<int>& proj_argmax, 
                           int idx,
                           V1I& knn_argmax_out )
{
  __TIC__(post_pred_clear)
  proj_unfold.assign( proj_unfold.size(),  0);
  proj_unfold2.assign( proj_unfold2.size(), 0);
  memset(knn_argmax_onehot.get(), 0, 21*unfold_width);
  __TOC__(post_pred_clear)

  __TIC__(unfold1)
  unfold(proj_range, proj_argmax, proj_unfold, proj_unfold2);  // 25x64x2048
  __TOC__(unfold1)

  __TIC__(unproj_rang)
  float tmp;
  float* k2_distances_p = k2_distances.get();
  for (auto i = 0; i < unfold_height; i++) { // 25
    auto offset = unfold_width * i; //  131072*i
    for (auto j = 0; j < pointsize[idx]; j++) {
      unproj_unfold_1_argmax[ offset+j ] = proj_unfold2[offset + idx_list[idx][j]];
      if (i!=12) {
         if((tmp = proj_unfold[offset + idx_list[idx][j]])>=0) {
            k2_distances_p[ offset+j ] = (abs(tmp - depth[idx][j]))*inv_gauss_k[i];
         } else {
            k2_distances_p[ offset+j ] = FLT_MAX*inv_gauss_k[i];
         }
      }
    }
  }
  __TOC__(unproj_rang)
 
  __TIC__(permute_topk)
  topk(  idx, k2_distances.get() , 5, knn_idx);
  __TOC__(permute_topk)

  __TIC__(gather)
  gather_1(k2_distances.get(), unproj_unfold_1_argmax, knn_idx, 5, pointsize[idx], knn_argmax);
  __TOC__(gather)

  __TIC__(scatter_add)
  scatter_add_(knn_argmax_onehot.get(), knn_argmax, 5, pointsize[idx]);
  __TOC__(scatter_add)

  __TIC__(argmax)
  argmax_with_map( knn_argmax_onehot.get(), (int)pointsize[idx], 21, knn_argmax_out);
  __TOC__(argmax)
}

}  // namespace ai
}  // namespace vitis

