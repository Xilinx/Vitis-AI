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

#include <cmath>
#include <algorithm>
#include <float.h>

// #include "caffe/common.hpp"
#include "caffe/util/quantize.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype fix_data(const Dtype x, const Dtype step,
    const Dtype lb, const Dtype ub) {
  return std::fmin(std::fmax(std::round(x/step)*step, lb), ub);
}

template float fix_data<float>(const float x, const float step,
    const float lb, const float ub);
template double fix_data<double>(const double x, const double step,
    const double lb, const double ub);

template <typename Dtype>
void caffe_cpu_fix(const int n, const Dtype* x, Dtype* y,
    const int bit_width, const int p) {
  Dtype step = std::pow(Dtype(2), -p);
  Dtype lower_bound = -std::pow(Dtype(2), bit_width-1)*step;
  Dtype upper_bound = std::pow(Dtype(2), bit_width-1)*step - step;
  for (auto i = 0; i < n; ++i) {
    y[i] = fix_data(x[i], step, lower_bound, upper_bound);
  }
}

template void caffe_cpu_fix<float>(const int n, const float* x, float* y,
    const int bit_width, const int p);
template void caffe_cpu_fix<double>(const int n, const double* x, double* y,
    const int bit_width, const int p);

template <typename Dtype>
Dtype fix_data1(const Dtype x, const Dtype step,
const Dtype lb, const Dtype ub) {
  Dtype tmp = x / step;
  // simulate DPU where to save hardware resource
  if ( tmp < 0 && ( tmp - floor( tmp ) ) == 0.5 )
    tmp = ceil( tmp );
  else
    tmp = round( tmp );
  return std::fmin(std::fmax(tmp * step, lb), ub);
}

template float fix_data1<float>(const float x, const float step,
    const float lb, const float ub);
template double fix_data1<double>(const double x, const double step,
    const double lb, const double ub);

template <typename Dtype>
void caffe_cpu_top_fix(const int n, const Dtype* x, Dtype* y,
    const int bit_width, const int p) {
  Dtype step = std::pow(Dtype(2), -p);
  Dtype lower_bound = -std::pow(Dtype(2), bit_width-1)*step;
  Dtype upper_bound = std::pow(Dtype(2), bit_width-1)*step - step;
  for (auto i = 0; i < n; ++i) {
    y[i] = fix_data1(x[i], step, lower_bound, upper_bound);
  }
}

template void caffe_cpu_top_fix<float>(const int n, const float* x, float* y,
    const int bit_width, const int p);
template void caffe_cpu_top_fix<double>(const int n, const double* x, double* y,
    const int bit_width, const int p);

int dpu_shift(const int x, const int s) {
  int y = 0;

  if (s >= 0) {
    // y = (abs(x) << s) * (x < 0 ? -1 : 1);
    y = x << s;
  } else {
    if (x >= 0) {
      y = (x + (1 << (-s-1))) >> (-s);
    } else {
      if ((-x) % (1 << (-s)) == (1 << (-s-1)))
        y = -((-x) >> (-s));
      else
        y = -(((-x) + (1 << (-s-1))) >> (-s));
    }
  }

  return y;
}

/*
template <>
void caffe_cpu_shift<int>(const int n, const int* a, const int b, int* y) {
  for (auto i = 0; i < n; ++i)
    y[i] = dpu_shift(a[i], b);
}
*/


template <typename Dtype>
Dtype caffe_cpu_fix_pos_overflow(const int n, const Dtype* x, const int bit_width){
  // Use half of step as a guard
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;

  // Find min and max value
  auto min_max = std::minmax_element(x, x + n);
  Dtype x_min = *min_max.first;
  Dtype x_max = *min_max.second;

  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0) {
    return SHRT_MAX;
  } else if(isnan(step)) {
    return SHRT_MIN;
  }
  return std::log2(1 / step);
}

template <typename Dtype>
Dtype caffe_cpu_fix_pos_diffs(const int n, const Dtype* x, const int bit_width, const int range) {
  // Calc search range for scale
  int max_scale;
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;
  auto min_max = std::minmax_element(x, x + n);
  Dtype x_min = *min_max.first;
  Dtype x_max = *min_max.second;

  // Find max_scale
  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0) {
    return SHRT_MAX;
  } else if(isnan(step)) {
    return SHRT_MIN;
  } else {
    max_scale = std::floor(std::log2(1 / step));
  }

  // Find fix pos in range [max_scale + range , max_scale]
  Dtype final_scale;
  final_scale = max_scale;
  Dtype fixed_diff_min = FLT_MAX;
  Dtype *buffer = new Dtype[n];
  for (int scale = max_scale; scale < max_scale + range; scale++) {
    caffe_cpu_fix<Dtype>(n, x, buffer, bit_width, scale);
    caffe_sub<Dtype>(n, x, buffer, buffer);
    caffe_powx<Dtype>(n, buffer, 2, buffer);
    Dtype fixed_diff = caffe_cpu_asum(n, buffer);
    if (fixed_diff < fixed_diff_min) {
      final_scale = scale;
      fixed_diff_min = fixed_diff;
    }
  }
  delete [] buffer;
  return final_scale;
}

// Diff_S_Sigmoid: minimize L2 norm of sigmoid(weights/activation) between fixed and float
template <typename Dtype>
Dtype caffe_cpu_fix_pos_diffs_sigmoid(const int n, const Dtype* x, const int bit_width, const int range) {
  // Calc search range for scale
  int max_scale;
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;
  auto min_max = std::minmax_element(x, x + n);
  Dtype x_min = *min_max.first;
  Dtype x_max = *min_max.second;

  // Find max_scale
  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0)
    max_scale = 0;
  else
    max_scale = std::floor(std::log2(1 / step));

  // Find fix pos in range [max_scale + range , max_scale]
  Dtype final_scale;
  final_scale = max_scale;
  Dtype fixed_diff_min = FLT_MAX;
  Dtype *sigmoid_x = new Dtype[n];
  Dtype *buffer    = new Dtype[n];
  for(int i=0; i<n; ++i){
    sigmoid_x[i] = 1.0f/(1.0f+exp(-x[i]));
  }
  for (int scale = max_scale; scale < max_scale + range; scale++) {
    caffe_cpu_fix<Dtype>(n, x, buffer, bit_width, scale);
    for(int j=0; j<n; ++j){
      buffer[j] = 1.0f/(1.0f+exp(-buffer[j]));
    }
    caffe_sub<Dtype>(n, sigmoid_x, buffer, buffer);
    caffe_powx<Dtype>(n, buffer, 2, buffer);
    Dtype fixed_diff = caffe_cpu_asum(n, buffer);
    if (fixed_diff < fixed_diff_min) {
      final_scale = scale;
      fixed_diff_min = fixed_diff;
    }
  }
  delete [] sigmoid_x;
  delete [] buffer;
  return final_scale;
}

template
float caffe_cpu_fix_pos_overflow(const int n, const float* x, const int bit_width);
template
float caffe_cpu_fix_pos_diffs(const int n, const float* x, const int bit_width, const int range);
template
float caffe_cpu_fix_pos_diffs_sigmoid(const int n, const float* x, const int bit_width, const int range);
template
double caffe_cpu_fix_pos_overflow(const int n, const double* x, const int bit_width);
template
double caffe_cpu_fix_pos_diffs(const int n, const double* x, const int bit_width, const int range);
template
double caffe_cpu_fix_pos_diffs_sigmoid(const int n, const double* x, const int bit_width, const int range);



template <typename Dtype>
Dtype caffe_fix_pos_overflow(const Blob<Dtype>& data, const int bit_width) {
  const int n = data.count();
#ifdef CPU_ONLY
  return caffe_cpu_fix_pos_overflow(n, data.cpu_data(), bit_width);
#else
  return caffe_gpu_fix_pos_overflow(n, data.gpu_data(), bit_width);
#endif
}

template <typename Dtype>
Dtype caffe_fix_pos_diffs(const Blob<Dtype>& data, const int bit_width, const int range) {
  const int n = data.count();
#ifdef CPU_ONLY
  return caffe_cpu_fix_pos_diffs(n, data.cpu_data(), bit_width, range);
#else
  return caffe_gpu_fix_pos_diffs(n, data.gpu_data(), bit_width, range);
#endif
}

template <typename Dtype>
Dtype caffe_fix_pos_diffs_sigmoid(const Blob<Dtype>& data, const int bit_width, const int range) {
  const int n = data.count();
#ifdef CPU_ONLY
  return caffe_cpu_fix_pos_diffs_sigmoid(n, data.cpu_data(), bit_width, range);
#else
  return caffe_gpu_fix_pos_diffs_sigmoid(n, data.gpu_data(), bit_width, range);
#endif
}

template <typename Dtype>
void caffe_fix(const Blob<Dtype>& data, Blob<Dtype>& data_fixed, const int bit_width, const int p) {
  const int n = data.count();
#ifdef CPU_ONLY
  caffe_cpu_fix(n, data.cpu_data(), data_fixed.mutable_cpu_data(), bit_width, p);
#else
  caffe_gpu_fix(n, data.gpu_data(), data_fixed.mutable_gpu_data(), bit_width, p);
#endif
}

template
float caffe_fix_pos_overflow(const Blob<float>& data, const int bit_width);
template
float caffe_fix_pos_diffs(const Blob<float>& data, const int bit_width, const int range);
template
float caffe_fix_pos_diffs_sigmoid(const Blob<float>& data, const int bit_width, const int range);
template
void caffe_fix(const Blob<float>& data, Blob<float>& data_fixed, const int bit_width, const int p);


template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype *x, Dtype *y, const int p) {
  Dtype step;
  if (p == SHRT_MAX) {
    step = 1;
  } else {
    step = std::pow(Dtype(2), p);
  }

  for (auto i = 0; i < n; ++i) {
    y[i] = x[i] * step;
  }
}

template void caffe_cpu_scale<float>(const int n, const float *x, float *y, const int p);
template void caffe_cpu_scale<double>(const int n, const double *x, double *y, const int p);

template <typename Dtype>
void caffe_cpu_trunc(const int n, const Dtype *x, Dtype *y, const int p) {
  Dtype scale = std::pow(Dtype(2), -p);
  for (auto i = 0; i < n; ++i) {
    y[i] = ( (int)(x[i] / scale) ) * scale;
  }
}

template void caffe_cpu_trunc<float>(const int n, const float *x, float *y, const int p);
template void caffe_cpu_trunc<double>(const int n, const double *x, double *y, const int p);

template <typename Dtype>
void caffe_cpu_pooling_scale(const int n, const Dtype *x, Dtype *y,  float scale) {
  for (auto i = 0; i < n; ++i) {
    y[i] = x[i] * scale;
  }
}

template void caffe_cpu_pooling_scale<float>(const int n, const float *x, float *y, float scale);
template void caffe_cpu_pooling_scale<double>(const int n, const double *x, double *y, float scale);
}  // namespace caffe
