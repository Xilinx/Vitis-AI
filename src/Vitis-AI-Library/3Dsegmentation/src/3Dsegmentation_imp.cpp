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
#include "post.hpp"
namespace vitis {
namespace ai {
using namespace std;
const float PI = 3.141592653589793;
const float proj_W = 2048.0;
const float proj_H = 64.0;
const float fov = 0.4886921905584123;
const float fov_down = -0.4363323129985824;
std::vector<int>  map_inv_{0, 10, 11, 15, 18,
                         20, 30, 31, 32, 40,
                         44, 48, 49, 50, 51,
                         70, 71, 72, 80, 81};
//const float fov_up = 0.05235987755982988;
template<typename T> 
void writefile(std::string& filename, std::vector<std::vector<std::vector<T>>>& data) {
  std::ofstream output_file(filename);
  for(size_t i = 0; i < data.size(); i++) { 
  for(size_t j = 0; j < data[i].size(); j++) {
  for(size_t k = 0; k < data[i][j].size(); k++) {
    output_file << data[i][j][k] << std::endl;
  }}}
}
template<typename T> 
void writefile(std::string& filename, std::vector<T>& data) {
  std::ofstream output_file(filename);
  for(size_t i = 0; i < data.size(); i++) 
    output_file << (int)data[i] << std::endl;
}
static void norm(std::vector<float>& input1, std::vector<float>& input2, std::vector<float>& input3,  std::vector<float>& output) {
  for(size_t i = 0; i < input1.size(); i++) {
    auto x = input1[i]; 
    auto y = input2[i]; 
    auto z = input3[i]; 
    output[i] = sqrt(x*x + y*y +z*z);
  }
}

float asinx(float x) {
  return (x * (1+x*x*(1.0/6.0+ x*x*(3.0/(2.0*4.0*5.0) + x*x*((1.0*3.0*5.0)/(2.0*4.0*6.0*7.0))))));
}

static void proj_y(std::vector<float>& z, std::vector<float>& depth, std::vector<float>& result) {
  for (size_t i = 0; i < z.size(); i++) {
    float t = asin(z[i] / depth[i]);
    t = floor((1.0 - (t + abs(fov_down)) / fov) * proj_H);
    result[i] = std::max(std::min(t, proj_H - 1.f), 0.f);
  }
}

/*static float atan2x(float y, float x) {
  float ax = std::abs(x), ay = std::abs(y);
  float a = std::min(ax, ay)/(std::max(ax, ay)+(float)DBL_EPSILON);
  float s = a*a;
  float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
  if(ay > ax) r = 1.57079637 - r;
  if(x < 0) r = 3.14159274f - r;
  if(y < 0) r = -r;
  return r;
}*/

static void proj_x(std::vector<float>& x, std::vector<float>& y, std::vector<float>& result) {
  for(size_t i = 0; i < x.size(); i++) {
    float t =-(atan2(y[i], x[i])); // cal yaw
    t = floor((0.5 *  (t / PI + 1.0)) * proj_W);
    result[i] = std::max(std::min(t, proj_W - 1.f), 0.f);
  }
}

template<typename T> 
static std::vector<int> argsort(const std::vector<T>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] > array[pos2]);});

	return array_index;
}

template<typename T> 
static void reorder(std::vector<T>& array, std::vector<int>& order) {
  std::vector<T> array_re(array.size());
  for(size_t i = 0; i < array.size(); i++) {
    array_re[i] = array[order[i]];
  }
  array.swap(array_re);
}

template<typename T1, typename T2> 
static void unreorder(std::vector<T1>& array, std::vector<T2>& order) {
  std::vector<T1> array_re(array.size());
  for(size_t i = 0; i < array.size(); i++) {
    array_re[order[i]] = array[i];
  }
  array.swap(array_re);
}

template<typename T> 
static void reorder2D(std::vector<T> in_array, std::vector<std::vector<T>>& out_array, std::vector<float>& order_x, std::vector<float>& order_y) {
  for(size_t i = 0; i < order_x.size(); i++) {
    out_array[order_y[i]][order_x[i]] = in_array[i];
  }
}

template<typename T1, typename T2> 
static void unreorder2D(std::vector<T1> in_array,  std::vector<T2>& out_array, std::vector<float>& order_x, std::vector<float>& order_y) {
  for(size_t i = 0; i < order_x.size(); i++) {
    out_array[i] = in_array[order_y[i]*2048 + order_x[i]];
    out_array[i] = map_inv_[out_array[i]];
  }
}

template<typename T>
void static permute(std::vector<std::vector<std::vector<T>>>& in_array, std::vector<int8_t>& out_array, int in_scale) {
  auto channels = in_array.size();
  auto height = in_array[0].size();
  auto width =  in_array[0][0].size(); 
  for(auto i = 0u; i < channels; i++) {
    for(auto j = 0u; j < height; j++) {
      for(auto k = 0u; k < width; k++ ) {
        //auto q = in_array[i][j][k] * in_scale;
        //if(q < -128 || q>127)
         // std::cout << q << " " << std::min(std::max((int)(in_array[i][j][k] * in_scale), -128), 127)<< std::endl;
        out_array[j * width * channels + k * channels + i] = (int8_t)std::min(std::max((int)(std::round(in_array[i][j][k] * in_scale)), -128), 127);
        //out_array[j * width * channels + k * channels + i] = std::max(std::min((int8_t)(in_array[i][j][k] * in_scale), (int8_t)127), (int8_t)-128);
      }
    }
  }
}

Segmentation3DImp::Segmentation3DImp(const std::string &model_name,
                               bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Segmentation3D>(model_name,
                                                   need_preprocess) {}

Segmentation3DImp::~Segmentation3DImp() {}
Segmentation3DResult Segmentation3DImp::run(std::vector<std::vector<float>>& array) {
  __TIC__(SEGMENTATION3D)
  __TIC__(PRE_PROCESS)
  __TIC__(get_depth)
  std::vector<float> depth(array[0].size());
  norm(array[0], array[1], array[2], depth);
  __TOC__(get_depth)
  auto unproj_range = depth;
  std::vector<float> py(array[2].size());
  std::vector<float> px(array[0].size());
  __TIC__(get_py)
  proj_y(array[2], depth, py);
  __TOC__(get_py)
  __TIC__(get_px)
  proj_x(array[0], array[1], px);
  __TOC__(get_px)
  auto un_px = px;
  auto un_py = py;
  std::vector<int> indices(depth.size());
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }
  __TIC__(reorder)
  auto order = argsort(depth);
  /*
  reorder(depth, order);
  reorder(indices, order);
  reorder(py, order);
  reorder(px, order);
  reorder(array[3], order);
  reorder(array[2], order);
  reorder(array[1], order);
  reorder(array[0], order);
  */
  auto unproj_n_points = array[0];
  __TOC__(reorder)
  __TIC__(reorder2d)
  std::vector<std::vector<std::vector<float>>> proj(5);
  for (auto & pr : proj) {
    pr.resize(64);
    for (auto & p : pr) {
      p.resize(2048, -1);
    }
  }
  std::vector<std::vector<int>> proj_idx(64);
  for (auto & p : proj_idx) {
    p.resize(2048, -1);
  }
  reorder2D(depth, proj[0], px, py);
  reorder2D(array[3], proj[4], px, py); //remission
  reorder2D(array[0], proj[1], px, py); //x
  reorder2D(array[1], proj[2], px, py); //y
  reorder2D(array[2], proj[3], px, py); //z
  reorder2D(indices, proj_idx, px, py); //idx
  std::vector<float> proj_range(proj[0].size() * proj[0][0].size());   //proj_range get
  for (auto i = 0u; i < proj[0].size(); i++) {
    for (auto j = 0u; j < proj[0][0].size(); j++) {
      proj_range[i * proj[0][0].size() + j] = proj[0][i][j];
    }
  }

  // proj_range = torch.from_numpy(scan.proj_range).clone()
  // unreorder(unproj_range,unproj_n_points);
  //  unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
  //
  __TOC__(reorder2d)
  __TIC__(mask_meanstd_permute)
  for (size_t i = 0u; i < proj.size(); i++) {
    for(size_t j = 0u; j < proj[0].size(); j++) {
      for(size_t k = 0u; k < proj[0][0].size(); k++) {
        proj[i][j][k] = (proj[i][j][k] - sensor_means_[i]) / sensor_stds_[i];
        proj[i][j][k] = proj_idx[j][k] <= 0? 0 : proj[i][j][k];
      }
    }
  }
  //  string proj_name = "proj.txt";
  //  writefile(proj_name, proj);
  std::vector<int8_t> pr_permute(5 * 64 * 2048);
  auto in_scale = vitis::ai::library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);
  //std::cout << in_scale << std::endl;
  permute(proj, pr_permute, in_scale);
  //string permute_name = "permute.txt";
  //writefile(permute_name, pr_permute);
  __TOC__(mask_meanstd_permute)
  __TOC__(PRE_PROCESS)
  __TIC__(SET_IMG)
  configurable_dpu_task_->setInputDataArray(pr_permute);
  __TOC__(SET_IMG)
  __TIC__(DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPU)
  auto t_size = configurable_dpu_task_->getOutputTensor()[0][0].height * configurable_dpu_task_->getOutputTensor()[0][0].width * configurable_dpu_task_->getOutputTensor()[0][0].channel;
  //std::vector<float> data_float(t_size);
  std::vector<int8_t> data_int(t_size);
  std::vector<float> proj_argmax;
  //float scale = vitis::ai::library::tensor_scale(configurable_dpu_task_->getOutputTensor()[0][0]);
  __TIC__(POST)
  memcpy(data_int.data(), ((int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(0)), sizeof(char) * t_size);
  //for (size_t i = 0; i < t_size; i++) {
  //  data_float[i] = scale * data_int[i];
 // }
  for (size_t i = 0; i < t_size; i = i + 20) {
    auto max_ind = std::max_element(data_int.data() + i, data_int.data() + i + 20);
    auto dis = std::distance(data_int.data() + i, max_ind);
    proj_argmax.push_back(dis);
  }
  //string argmax_name = "argmax.txt";
  //writefile(argmax_name, proj_argmax);
  std::vector<int> unproj_argmax;
    bool enable_knn = configurable_dpu_task_->getConfig().segmentation_3d_param().enable_knn();
  if(!enable_knn) {
    unproj_argmax.resize(un_px.size());
    unreorder2D(proj_argmax, unproj_argmax, un_px, un_py);
  } else {
    unproj_argmax = vitis::ai::Segmentation3DPost::post_prec(proj_range, proj_argmax, unproj_range, un_px, un_py);
    for (auto i = 0u; i < unproj_argmax.size(); i++) {
      unproj_argmax[i] = map_inv_[unproj_argmax[i]];
    }
  }
  __TOC__(POST)
  Segmentation3DResult rs{getInputWidth(), getInputHeight(), unproj_argmax};
  rs.array.swap(unproj_argmax);
  __TOC__(SEGMENTATION3D)
  return rs;
}


std::vector<Segmentation3DResult> Segmentation3DImp::run(std::vector<std::vector<std::vector<float>>>& arrays) {
  auto batch_size = std::min(get_input_batch(), arrays.size());
  std::vector<std::vector<int8_t>> pr_permutes(batch_size);
  std::vector<std::vector<float>> unproj_ranges(batch_size); 
  std::vector<std::vector<float>> proj_ranges(batch_size); 
  std::vector<std::vector<float>> unproj_n_points(batch_size);
  for(auto & pr_permute : pr_permutes)
    pr_permute.resize(5 * 64 * 2048);
  std::vector<std::vector<float>> un_pxs(batch_size);
  std::vector<std::vector<float>> un_pys(batch_size);
  for (auto batch_ind = 0u; batch_ind < batch_size; batch_ind++) {
    auto array = arrays[batch_ind];
    __TIC__(PRE_PROCESS)
    __TIC__(get_depth)
    std::vector<float> depth(array[0].size());
    norm(array[0], array[1], array[2], depth);
    unproj_ranges[batch_ind] = depth;
    __TOC__(get_depth)
    std::vector<float> py(array[2].size());
    std::vector<float> px(array[0].size());
    __TIC__(get_py)
    proj_y(array[2], depth, py);
    __TOC__(get_py)
    __TIC__(get_px)
    proj_x(array[0], array[1], px);
    __TOC__(get_px)
    un_pxs[batch_ind] = px;
    un_pys[batch_ind] = py;
    std::vector<int> indices(depth.size());
    for (size_t i = 0; i < indices.size(); i++) {
      indices[i] = i;
    }
    __TIC__(reorder)
    auto order = argsort(depth);
    /*
    reorder(depth, order);
    reorder(indices, order);
    reorder(py, order);
    reorder(px, order);
    reorder(array[3], order);
    reorder(array[2], order);
    reorder(array[1], order);
    reorder(array[0], order);
    */
    unproj_n_points[batch_ind] = array[0];
    __TOC__(reorder)
    __TIC__(reorder2d)
    std::vector<std::vector<std::vector<float>>> proj(5);
    for (auto & pr : proj) {
      pr.resize(64);
      for (auto & p : pr) {
        p.resize(2048, -1);
      }
    }
    std::vector<std::vector<int>> proj_idx(64);
    for (auto & p : proj_idx) {
      p.resize(2048, -1);
    }
    reorder2D(depth, proj[0], px, py);
    reorder2D(array[3], proj[4], px, py); //remission
    reorder2D(array[0], proj[1], px, py); //x
    reorder2D(array[1], proj[2], px, py); //y
    reorder2D(array[2], proj[3], px, py); //z
    reorder2D(indices, proj_idx, px, py); //idx
    proj_ranges[batch_ind].resize(proj[0].size() * proj[0][0].size());   //proj_range get
    for (auto i = 0u; i < proj[0].size(); i++) {
      for (auto j = 0u; j < proj[0][0].size(); j++) {
        proj_ranges[batch_ind][i * proj[0][0].size() + j] = proj[0][i][j];
      }
    }

    // proj_range = torch.from_numpy(scan.proj_range).clone()
    //unreorder(unproj_ranges[batch_ind],unproj_n_points[batch_ind]);
    __TOC__(reorder2d)
    __TIC__(mask_meanstd_permute)
    for (size_t i = 0u; i < proj.size(); i++) {
      for(size_t j = 0u; j < proj[0].size(); j++) {
        for(size_t k = 0u; k < proj[0][0].size(); k++) {
          proj[i][j][k] = (proj[i][j][k] - sensor_means_[i]) / sensor_stds_[i];
          proj[i][j][k] = proj_idx[j][k] <= 0? 0 : proj[i][j][k];
        }
      }
    }
    auto in_scale = vitis::ai::library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][0]);

    permute(proj, pr_permutes[batch_ind], in_scale);
    __TOC__(mask_meanstd_permute)
    __TOC__(PRE_PROCESS)
  }
  __TIC__(SET_IMG)
  configurable_dpu_task_->setInputDataArray(pr_permutes);
  __TOC__(SET_IMG)
  __TIC__(DPU)
  configurable_dpu_task_->run(0);
  __TOC__(DPU)
  __TIC__(POST)
  std::vector<std::vector<int>> unproj_argmaxs(batch_size);
  for(auto batch_ind = 0u; batch_ind < batch_size; batch_ind++)
    unproj_argmaxs[batch_ind].resize(un_pxs[batch_ind].size());
  for (size_t batch_ind = 0; batch_ind < batch_size; batch_ind++) {
    auto t_size = configurable_dpu_task_->getOutputTensor()[0][0].size/configurable_dpu_task_->getOutputTensor()[0][0].batch;
    std::vector<float> data_float(t_size);
    std::vector<int8_t> data_int(t_size);
    std::vector<float> proj_argmax;
    float scale = vitis::ai::library::tensor_scale(configurable_dpu_task_->getOutputTensor()[0][0]);
    memcpy(data_int.data(), ((int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(batch_ind)), sizeof(char) * t_size);
    for (size_t i = 0; i < t_size; i++) {
      data_float[i] = scale * data_int[i];
    }
    for (size_t i = 0; i < t_size; i = i + 20) {
      auto max_ind = std::max_element(data_float.data() + i, data_float.data() + i + 20);
      auto dis = std::distance(data_float.data() + i, max_ind);
      proj_argmax.push_back(dis);
    }
    bool enable_knn = configurable_dpu_task_->getConfig().segmentation_3d_param().enable_knn();
    if(!enable_knn) {
      unproj_argmaxs[batch_ind].resize(un_pxs[batch_ind].size());
      unreorder2D(proj_argmax, unproj_argmaxs[batch_ind], un_pxs[batch_ind], un_pys[batch_ind]);
    } else {
      unproj_argmaxs[batch_ind] = vitis::ai::Segmentation3DPost::post_prec(proj_ranges[batch_ind], proj_argmax, unproj_ranges[batch_ind], un_pxs[batch_ind], un_pys[batch_ind]);
      for (auto i = 0u; i < unproj_argmaxs[batch_ind].size(); i++) {
        unproj_argmaxs[batch_ind][i] = map_inv_[unproj_argmaxs[batch_ind][i]];
      }
  }
  }
  std::vector<Segmentation3DResult> all_rs(batch_size);
  for(size_t batch_ind = 0; batch_ind < batch_size; batch_ind++) 
    all_rs[batch_ind] = Segmentation3DResult{getInputWidth(), getInputHeight(), unproj_argmaxs[batch_ind]};
  __TOC__(POST)
  return all_rs;
}



}  // namespace ai
}  // namespace vitis
