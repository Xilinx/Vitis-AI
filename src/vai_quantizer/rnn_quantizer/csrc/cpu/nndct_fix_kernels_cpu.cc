

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

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
//#include <iostream>
#include "../../include/cpu/nndct_fix_kernels_cpu.h"
#include "../../include/cpu/nndct_cpu_math.h"


template<typename Dtype>
static void _fix_neuron_v1(const int N, 
                           const Dtype* src, 
                           const Dtype* fragpos, 
                           Dtype* dst, 
                           int val_max, 
                           int keep_scale, 
                           int method){
  //NNDCT_KERNEL_LOOP(index, N)
  for(int index=0; index < N; index ++){
    //method:
    //1: dummy
    //2: for CNN feature map
    //3: for weights and bias
    //4: for RNN feature map
    int result_ = 0;
    Dtype val_amp = pow(2, *fragpos);
    _fix_neuron_v2_cpu(src[index], 
                       result_, 
                       val_max, 
                       val_amp, 
                       method);
    if(0 != keep_scale)
      dst[index] = Dtype(result_) * (1 / val_amp);
    else
      dst[index] = result_;
  }
}

template<typename Dtype>
static void _fix_neuron_v2(const int N, 
                           const Dtype* src, 
                           Dtype* dst, 
                           int val_max, 
                           Dtype val_amp, 
                           int keep_scale, 
                           int method){
  //NNDCT_KERNEL_LOOP(index, N){
  for(int index=0; index < N; index ++){
    //method: 
    //1: dummy
    //2: for CNN feature map
    //3: for weights and bias
    //4: for RNN feature map
    int result_ = 0;
    _fix_neuron_v2_cpu(src[index], 
                       result_, 
                       val_max, 
                       val_amp, 
                       method);
    if(0 != keep_scale)
      dst[index] = Dtype(result_) * (1 / val_amp);
    else
      dst[index] = result_;
    
  }
}

template<typename Dtype>
void cpu_fix_neuron_v1(const int N, 
                       const Dtype* src, 
                       const Dtype* fragpos, 
                       Dtype* dst, 
                       int val_max, 
                       int keep_scale, 
                       int method){
  _fix_neuron_v1(
    N, 
    src, 
    fragpos, 
    dst, 
    val_max, 
    keep_scale,
    method);
}

template
void cpu_fix_neuron_v1<float>(const int N, 
                               const float* src,
                               const float* fragpos, 
                               float* dst, 
                               int val_max, 
                               int keep_scale, 
                               int method);
template
void cpu_fix_neuron_v1<double>(const int N, 
                                const double* src,
                                const double* fragpos, 
                                double* dst, 
                                int val_max, 
                                int keep_scale, 
                                int method);
template<typename Dtype>
void cpu_fix_neuron_v2(const int N, 
                        const Dtype* src, 
                        Dtype* dst, 
                        int val_max, 
                        Dtype val_amp, 
                        int keep_scale, 
                        int method){
  _fix_neuron_v2(
    N, 
    src, 
    dst, 
    val_max, 
    val_amp, 
    keep_scale,
    method);
}

template
void cpu_fix_neuron_v2<float>(const int N, 
                               const float* src,
                               float* dst, 
                               int val_max, 
                               float val_amp, 
                               int keep_scale, 
                               int method);
template
void cpu_fix_neuron_v2<double>(const int N, 
                                const double* src,
                                double* dst, 
                                int val_max, 
                                double val_amp, 
                                int keep_scale, 
                                int method);

template<typename Dtype> 
void cpu_diff_S(const int N, 
                const Dtype* src,
                Dtype* buffer,
                Dtype* output, 
                int bitwidth,
                int range,
                int method){
  // Calc search range for scale
  int max_scale;
  Dtype fix_lb = -pow(2, bitwidth - 1) - 0.5;
  Dtype fix_ub = pow(2, bitwidth - 1) - 0.5;
  
  Dtype x_max, x_min;
  cpu_max(N, src, x_max);
  cpu_min(N, src, x_min);
  
  // Find max_scale
  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  //Dtype step = max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0) {
    max_scale = 0;
  } else {
    max_scale = floor(log2(1 / step));
  }
#if 0
  printf( "$$$$$$$$$$$ bw: %d range: %d method: %d\n",
         bitwidth, range, method ); 
  printf( "$$$$$$$$$$$ max: %g min: %g\n",
         x_max, x_min );
  printf( "$$$$$$$$$$$ overflow scale is %d\n", max_scale );
#endif

  // Find fix pos in range [max_scale + range , max_scale]
  Dtype final_scale = max_scale;
  Dtype fixed_diff_min = FLT_MAX;
  for (int scale = max_scale; scale < max_scale + range; scale++) {
    cpu_fix_neuron_v2(N,
                      src,
                      buffer,
                      1<<(bitwidth-1),
                      Dtype(pow(2, scale)),
                      1,
                      method);
    cpu_sub(N, src, buffer);
    cpu_pow(N, buffer, Dtype(2));
    Dtype fixed_diff;
    cpu_sum(N, buffer, fixed_diff);
    if (fixed_diff < fixed_diff_min) {
      final_scale = scale;
      fixed_diff_min = fixed_diff;
    }
  }
  //final_scale = final_scale > 15 ? 15: final_scale;
  cpu_set(1, output, final_scale);
#if 0
  printf( "$$$$$$$$$$$ diffs scale is %g, setting to %p...\n", 
          final_scale,
          output ); fflush(stdout);
#endif
}

template 
void cpu_diff_S<float>(const int N, 
                        const float* src, 
                        float* buffer, 
                        float* output, 
                        int bitwidth, 
                        int range, 
                        int method);
template 
void cpu_diff_S<double>(const int N, 
                         const double* src, 
                         double* buffer, 
                         double* output, 
                         int bitwidth, 
                         int range, 
                         int method);

template<typename Dtype>
static void _sigmoid_table_lookup(const int N,
                                  const int fragpos,
                                  const Dtype scale,
                                  const Dtype fuzz,
                                  const Dtype* input,
                                  const Dtype* table,
                                  Dtype* output) {
  //NNDCT_KERNEL_LOOP(i, N){
  for(int i=0; i < N; i++){
    if (input[i] >= 8.0) 
      output[i] = 1.0 - fuzz;
    else if (input[i] < -8.0)
      output[i] = 0.0;
    else {
      int x = int(input[i] * scale);
      int pos = 0;
      if (x >= 0) {
        if (fragpos >= 7)
          pos = (x >> (fragpos - 7)) % 1024;
        else
          pos = (x << (7 - fragpos)) % 1024;
        output[i] = table[pos + 1024] * fuzz;
      }
      else {
        //if (fragpos >= 7)
        //  pos = (abs(x) >> (fragpos - 7)) % 1024;
        //else
        //  pos = (x << (7 - fragpos)) % 1024;
        pos = abs(int(floor(x / pow(2.0, (fragpos - 7))))) % 1024;
        if (x >> fragpos == -8 && pos == 0)
          output[i] = table[pos] * fuzz;
        else
          output[i] = table[1024 - pos] * fuzz;
      }
    }
  }
}

template<typename Dtype>
void cpu_sigmoid_table_lookup(const int N, 
                              const Dtype* input, 
                              const Dtype* table,
                              Dtype* output,
                              int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _sigmoid_table_lookup(
      N, 
      fragpos, 
      scale, 
      fuzz, 
      input, 
      table, 
      output);
} 

template
void cpu_sigmoid_table_lookup<float>(const int N, 
                                      const float* input, 
                                      const float* table,
                                      float* output,
                                      int fragpos);
template
void cpu_sigmoid_table_lookup<double>(const int N, 
                                       const double* input, 
                                       const double* table,
                                       double* output,
                                       int fragpos);

template<typename Dtype>
static void _tanh_table_lookup(const int N,
                               const int fragpos,
                               const Dtype scale,
                               const Dtype fuzz,
                               const Dtype* input,
                               const Dtype* table,
                               Dtype* output) {
  //NNDCT_KERNEL_LOOP(i, N){
  for(int i=0; i < N; i++){
    if (input[i] >= 4.0) 
      output[i] = 1.0 - fuzz;
    else if (input[i] < -4.0)
      output[i] = -1.0;
    else {
      int x = int(input[i] * scale);
      int pos = 0;
      if (x >= 0) {
        if (fragpos >= 8)
          pos = (x >> (fragpos - 8)) % 1024;
        else
          pos = (x << (8 - fragpos)) % 1024;
        output[i] = table[pos + 1024] * fuzz;
      }
      else {
        //if (fragpos >= 8)
        //  pos = (abs(x) >> (fragpos - 8)) % 1024;
        //else
        //  pos = (abs(x) << (8 - fragpos)) % 1024;
        pos = abs(int(floor(x / pow(2.0, (fragpos - 8))))) % 1024;
        if (x >> fragpos == -4 && pos == 0)
          output[i] = table[pos] * fuzz;
        else
          output[i] = table[1024 - pos] * fuzz;
      }
    }
  }
}

template<typename Dtype>
void cpu_tanh_table_lookup(const int N, 
                            const Dtype* input, 
                            const Dtype* table,
                            Dtype* output,
                            int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _tanh_table_lookup(
      N, 
      fragpos, 
      scale, 
      fuzz, 
      input, 
      table, 
      output);
} 

template
void cpu_tanh_table_lookup<float>(const int N, 
                                   const float* input, 
                                   const float* table,
                                   float* output,
                                   int fragpos);
template
void cpu_tanh_table_lookup<double>(const int N, 
                                    const double* input, 
                                    const double* table,
                                    double* output,
                                    int fragpos);
