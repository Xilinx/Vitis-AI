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

#ifndef CAFFE_UTIL_QUANTIZE_HPP_
#define CAFFE_UTIL_QUANTIZE_HPP_

#include "caffe/blob.hpp"

namespace caffe {


// p is position of the dot
template <typename Dtype>
void caffe_cpu_fix(const int n, const Dtype* x, Dtype* y, const int bit_width, const int p);

// wrapper functions
template <typename Dtype>
Dtype caffe_fix_pos_overflow(const Blob<Dtype>& data, const int bit_width);
template <typename Dtype>
Dtype caffe_fix_pos_diffs(const Blob<Dtype>& data, const int bit_width, const int range = 5);
template <typename Dtype>
Dtype caffe_fix_pos_diffs_sigmoid(const Blob<Dtype>& data, const int bit_width, const int range = 5);
template <typename Dtype>
void caffe_fix(const Blob<Dtype>& data, Blob<Dtype>& data_fixed, const int bit_width, const int p);

// cpu functions
template <typename Dtype>
Dtype caffe_cpu_fix_pos_overflow(const int n, const Dtype* x, const int bit_width);
template <typename Dtype>
Dtype caffe_cpu_fix_pos_diffs(const int n, const Dtype* x, const int bit_width, const int range = 5);
template <typename Dtype>
Dtype caffe_cpu_fix_pos_diffs_sigmoid(const int n, const Dtype* x, const int bit_width, const int range = 5);

template <typename Dtype>
void caffe_cpu_top_fix(const int n, const Dtype* x, Dtype* y,
    const int bit_width, const int p);

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype* x, Dtype* y, const int p);

template <typename Dtype>
void caffe_cpu_trunc(const int n, const Dtype* x, Dtype* y, const int p);

template <typename Dtype>
void caffe_cpu_pooling_scale(const int n, const Dtype* x, Dtype* y, float scale);

#ifndef CPU_ONLY
// gpu functions
template <typename Dtype>
Dtype caffe_gpu_fix_pos_overflow(const int n, const Dtype* x, const int bit_width);
template <typename Dtype>
Dtype caffe_gpu_fix_pos_diffs(const int n, const Dtype* x, const int bit_width, const int range = 5);
template <typename Dtype>
Dtype caffe_gpu_fix_pos_diffs_sigmoid(const int n, const Dtype* x, const int bit_width, const int range = 5);
//template <typename Dtype>
//Dtype caffe_gpu_fix_pos_diffa(const int n, const Dtype* x, const int bit_width, const int range = 5);
template <typename Dtype>
  void caffe_gpu_fix(const int n, const Dtype* x, Dtype* y, const int bit_width, const int p);

template <typename Dtype>
void caffe_gpu_top_fix(const int n, const Dtype* x, Dtype* y,
    const int bit_width, const int p);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype* x, Dtype* y, const int p);

template <typename Dtype>
void caffe_gpu_trunc(const int n, const Dtype* x, Dtype* y, const int p);

template <typename Dtype>
void caffe_pooling_scale(const int n, const Dtype* x, Dtype* y, float scale);

//template <typename Dtype>
//void caffe_gpu_fix_overflow(const int n, const Dtype* x, Dtype* y,
//    const int bit_level, const int max_scale, const int min_scale, int& final_scale);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_QUANTIZE_HPP_
