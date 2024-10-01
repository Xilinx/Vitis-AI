

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

#include <stdio.h>
#include <stdlib.h>
//#include <iostream>
#include <math.h>
#include <algorithm>
#include <float.h>
//#include <math_constants.h>
#include "../../include/cpu/nndct_cpu_math.h"


template<typename Dtype>
void cpu_set(const int N, Dtype* data, Dtype val){
  for(int index=0; index<N; index++){
    data[index] = val;
  }
}
template 
void cpu_set<float>(const int N, float* data, float val);
template 
void cpu_set<double>(const int N, double* data, double val);
  

template<typename Dtype>
void cpu_scale_inplace(const int N, Dtype* data, Dtype scale){
  for(int index=0; index<N; index++){
    data[index] *= scale;
  }
}
template
void cpu_scale_inplace<float>(const int N, float* data, float scale);
template
void cpu_scale_inplace<double>(const int N, double* data, double scale);


template<typename Dtype>
void cpu_scale(const int N, const Dtype* src, Dtype* dst, Dtype scale){
  for(int index=0; index<N; index++){
    dst[index] = scale * src[index];
  }
}
template
void cpu_scale<float>(const int N, const float* src, float* dst, float scale);
template
void cpu_scale<double>(const int N, const double* src, double* dst, double scale);


template<typename Dtype>
void cpu_pow(const int N, Dtype* data, Dtype power){
  for(int index=0; index < N; index ++){
    data[index] = pow(data[index], power);
  }
}
template
void cpu_pow<float>(const int N, float* data, float power);
template
void cpu_pow<double>(const int N, double* data, double power);


template<typename Dtype>
void cpu_max(const int N, const Dtype* src, Dtype& dst){
  dst = src[0];
  for(int i = 1; i < N; i ++){
    Dtype tmp = src[i];
    if(dst < tmp){
      dst = tmp;
    }
  }
}
template
void cpu_max<float>(const int N, const float* src, float& dst);
template
void cpu_max<double>(const int N, const double* src, double& dst);


template<typename Dtype>
void cpu_min(const int N, const Dtype* src, Dtype& dst){
  dst = src[0];
  for(int i = 1; i < N; i ++){
    Dtype tmp = src[i];
    if(dst > tmp){
      dst = tmp;
    }
  }
}
template
void cpu_min<float>(const int N, const float* src, float& dst);
template
void cpu_min<double>(const int N, const double* src, double& dst);


template<typename Dtype>
void cpu_sum(const int N, const Dtype* src, Dtype& dst){
  dst = Dtype(0);
  for(int index=0; index<N; index++){
    dst = dst + src[index];
  }
}
template
void cpu_sum<float>(const int N, const float* src, float& dst);
template
void cpu_sum<double>(const int N, const double* src, double& dst);


template<typename Dtype>
void cpu_sub(const int N, const Dtype* src, Dtype* dst){
  for(int index=0; index < N; index++){
    //Dtype tmp = src[index] - dst[index];
    dst[index] = src[index] - dst[index];
  }
}
template
void cpu_sub<float>(const int N, const float* src, float* dst);
template
void cpu_sub<double>(const int N, const double* src, double* dst);
