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
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <cassert>
#include <initializer_list>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis { namespace ai {
namespace centerpoint{

constexpr uint32_t MAX_POINTS_NUM = 40;
constexpr uint32_t MAX_VOXELS_NUM = 20000; 

template<typename T>
struct DataContainer {
  std::vector<uint32_t> shape;
  std::vector<T> data; 
  std::vector<uint64_t> stride;
  
  explicit DataContainer(const std::vector<uint32_t>& data_shape, const T& val)
               : shape{data_shape}, stride(data_shape.size()) {
    long long size = 1;
    // e.g data_shape = {3, 4, 5} 
    //     stride = {20, 5, 1}
    for (auto i = 0u; i < data_shape.size(); ++i) {
      stride[data_shape.size() - 1 - i] = size;
      size *= data_shape[data_shape.size() - 1 - i];
    }
    data.assign(size, val);
  }

  DataContainer(const DataContainer& other) = default;
  DataContainer(DataContainer&& other) = default;
  DataContainer& operator = (const DataContainer& other){
    this->shape = other.shape;
    this->data = other.data;
    this->stride = other.stride;
    return *this;
  }

  DataContainer& operator = (DataContainer&& other){
    this->shape = std::move(other.shape);
    this->data = std::move(other.data);
    this->stride = std::move(other.stride);
    return *this;
  }

  virtual ~DataContainer() = default;

  T& at(std::initializer_list<int32_t> l) {
    assert(l.size() == shape.size());
    auto pos = 0llu; 
    auto it = l.begin();
    for (uint32_t i = 0u; i < stride.size(); ++i, ++it) {
      pos += stride[i] * (*it); 
    } 
    assert(pos < data.size()); 
    return data[pos];
  }

  const T& at(std::initializer_list<int32_t> l) const {
    assert(l.size() == shape.size());
    auto pos = 0llu; 
    auto it = l.begin();
    for (uint32_t i = 0u; i < stride.size(); ++i, ++it) {
      pos += stride[i] * (*it); 
    } 
    assert(pos < data.size()); 
    return data[pos];
  }

};

typedef struct {
  int voxel_num;
  DataContainer<float> voxels;
  DataContainer<int> num_points; // points number of every voxel
  DataContainer<int> coors;
} VoxelizeResult;

typedef struct {
  int voxel_num;
  vector<float> voxels;
  vector<int> num_points; // points number of every voxel
  vector<int> coors;
} VoxelizeResult2;


// note: need input tensor size
std::vector<int> preprocess(const DataContainer<float> &points, int8_t *input_tensor_ptr, float input_tensor_scale);

std::vector<int> preprocess2(const vector<float> &points, int8_t *input_tensor_ptr, float input_tensor_scale);
//std::vector<int> preprocess3(const vector<float> &points, int8_t *input_tensor_ptr, float input_tensor_scale);
std::vector<int> preprocess3(const vector<float> &points, int dim, 
                             const std::vector<float> &input_mean, const std::vector<float> &input_scale, 
                             int8_t *input_tensor_ptr);
} // end of pointpillars_nuscenes
}}


