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
#include "post.hpp"
#include "post_util.hpp"
#include "float.h"

namespace vitis {
namespace ai {
namespace Segmentation3DPost {
using namespace std;
#define VNAME(value) (#value)
int W = 2048;
int center = 12;
float cutoff = 1.0;
float nclasses = 20;

std::vector<float> inv_gauss_k{0.9970, 0.9867, 0.9781, 0.9867, 0.9970, 0.9867, 0.9404
                    , 0.9017, 0.9404, 0.9867, 0.9781, 0.9017, 0.8379, 0.9017
                    , 0.9781, 0.9867, 0.9404, 0.9017, 0.9404, 0.9867, 0.9970
                    , 0.9867, 0.9781, 0.9867, 0.9970};
template <typename T>
void pt(std::vector<T> input, int a, int b) {
  for (auto i = a; i < b; i++)
    cout << VNAME(input) << "  idx:  " << i << "\t" << input[i] << endl;
}



std::vector<int> post_prec(std::vector<float> proj_range,  std::vector<float> proj_argmax, std::vector<float> unproj_range, std::vector<float> px, std::vector<float> py) { 
  vector<float> proj_unfold;
  unfold(proj_range, vector<int> {1,1,64,2048}, proj_unfold,  make_pair(5, 5), make_pair(1, 1), make_pair(2, 2), make_pair(1, 1));
  //cout <<  "unfold input/output size " << proj_range.size() << "/" << proj_unfold.size() << endl; 
  //get idx list   idx_list = py * W + px
  vector<int> idx_list(px.size());
  for (auto i = 0u; i < idx_list.size(); i++) {
    idx_list[i] = py[i] * W + px[i];
  }
  size_t unfold_height = 25;
  size_t unfold_width = 131072;
  //unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list] unproj_unfold_k_rang.shape  torch.Size([1, 25, 123389])   proj_unfold_k_rang.shape  torch.Size([1, 25, 131072])
  std::vector<float> unproj_unfold_k_rang;
  for (auto i = 0u; i < unfold_height; i++) {
    auto offset = unfold_width * i;
    vector<float> temp(idx_list.size());
    for (auto j = 0u; j < idx_list.size(); j++) {
      auto tmp = proj_unfold[offset + idx_list[j]];
      (tmp < 0 ) ? temp[j] = FLT_MAX * 1: temp[j] = tmp;
    }
    unproj_unfold_k_rang.insert(unproj_unfold_k_rang.end(), temp.begin(), temp.end());
  }
  auto start = unproj_unfold_k_rang.begin() + (center)* idx_list.size();
  std::copy(unproj_range.begin(), unproj_range.end(), start);

  std::vector<float> k2_distances(unproj_unfold_k_rang.size());
  //cout << "k2_dis size " << k2_distances.size() << " "  << unproj_unfold_k_rang.size() << " " << proj_range.size()<< endl;
  for (auto i = 0u; i < k2_distances.size(); i++) {
    k2_distances[i] = abs(unproj_unfold_k_rang[i] - unproj_range[i % unproj_range.size()]);
  }

  //pt(k2_distances, 2480668, 2480888);
  //cout << "k2_dis size " << k2_distances.size() << endl;
  for(auto i = 0u; i < inv_gauss_k.size(); i++) {
    auto offset = i*idx_list.size();
    for(auto j = 0u; j < idx_list.size(); j++) {
      auto temp = k2_distances[offset + j] * inv_gauss_k[i];
      k2_distances[offset + j] = temp;
    }
  }
  //pt(k2_distances, 2980668, 2980888);
  auto k2_permute = permute(k2_distances, 1, unfold_height, idx_list.size());
  std::vector<float> knn_value;
  std::vector<uint32_t> knn_idx_pm;
  for (auto i = 0u; i < idx_list.size(); i++) {
    auto offset = i*unfold_height;
    auto st = k2_permute.begin() + offset;
    auto ed = k2_permute.begin() + offset + unfold_height; 
    std::vector<float> k2tmp(st, ed);
    auto topktmp = topK_distance(k2tmp, 5);
    knn_value.insert(knn_value.end(), topktmp.first.begin(), topktmp.first.end());
    knn_idx_pm.insert(knn_idx_pm.end(), topktmp.second.begin(), topktmp.second.end());
  }
  auto knn_idx = permute(knn_idx_pm, 1, idx_list.size(), 5);
  std::vector<float>  proj_unfold_1_argmax;
  unfold(proj_argmax, vector<int> {1,1,64,2048}, proj_unfold_1_argmax,  make_pair(5, 5), make_pair(1, 1), make_pair(2, 2), make_pair(1, 1));
  std::vector<float> unproj_unfold_1_argmax;
  for (auto i = 0u; i < unfold_height; i++) {
    auto offset = unfold_width * i;
    vector<float> temp(idx_list.size());
    for (auto j = 0u; j < idx_list.size(); j++) {
      temp[j] = proj_unfold_1_argmax[offset + idx_list[j]];
    }
    unproj_unfold_1_argmax.insert(unproj_unfold_1_argmax.end(), temp.begin(), temp.end());
  }
  
  auto knn_argmax = gather_1(unproj_unfold_1_argmax, knn_idx, 1, 5, idx_list.size());
  auto knn_distance = gather_1(k2_distances, knn_idx, 1, 5, idx_list.size());

  //pt(knn_distance, 214600, 215800);
  //pt(knn_argmax, 214600, 215800);
  for (auto i = 0u; i < knn_distance.size(); i++) {
    if (knn_distance[i] > cutoff) {
      knn_argmax[i] = nclasses;
    }
  }
  std::vector<float> knn_argmax_onehot((nclasses + 1) * idx_list.size(), 0); 
  std::vector<float> ones(knn_argmax.size(), 1);
  scatter_add_(knn_argmax_onehot, knn_argmax, ones, std::vector<int>{1, 5, (int)idx_list.size()});
  //pt(knn_argmax_onehot, 1883406, 1883406 + 400);
	auto kaop = permute(knn_argmax_onehot, 1, 21, (int)idx_list.size());
	auto knn_argmax_out = argmax(kaop, std::vector<int>{1, (int)idx_list.size(), 21});  
  return knn_argmax_out;
}


} //namespace
}
}
