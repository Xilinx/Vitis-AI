

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
//#include <bitset>
#include "../../include/cpu/nndct_fix_kernels_cpu.h"
#include "../../include/cpu/nndct_cpu_math.h"

template<typename Dtype>
static void _cpu_vai_round(const int N, 
                           const Dtype* src, 
                           Dtype* dst, 
                           int method){
  for(int index=0; index < N; index ++){
    int result_ = 0;
    _vai_round_cpu(src[index], result_, method);
    dst[index] = result_;
  }
}

template<typename Dtype>
void cpu_vai_round(const int N, 
                   const Dtype* src, 
                   Dtype* dst, 
                   int method){
  _cpu_vai_round(N, 
                 src, 
                 dst, 
                 method);
}

template
void cpu_vai_round<float>(const int N, 
                          const float* src, 
                          float* dst, 
                          int method);

template
void cpu_vai_round<double>(const int N, 
                           const double* src, 
                           double* dst, 
                           int method);

template<typename Dtype>
static void _fix_neuron_v1(const int N, 
                           const Dtype* src, 
                           const Dtype* fragpos, 
                           Dtype* dst, 
                           int val_min,
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
                       val_min,
                       val_max, 
                       val_amp, 
                       0,
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
                           int val_min,
                           int val_max, 
                           Dtype val_amp, 
                           int zero_point,
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
                       val_min,
                       val_max, 
                       val_amp, 
                       zero_point,
                       method);
    if(0 != keep_scale)
      dst[index] = (Dtype(result_)-Dtype(zero_point)) * (1 / val_amp);
    else
      dst[index] = result_;
  }
}

template<typename Dtype>
void cpu_fix_neuron_v1(const int N, 
                       const Dtype* src, 
                       const Dtype* fragpos, 
                       Dtype* dst, 
                       int val_min,
                       int val_max, 
                       int keep_scale, 
                       int method){
  _fix_neuron_v1(
    N, 
    src, 
    fragpos, 
    dst, 
    val_min,
    val_max, 
    keep_scale,
    method);
}

template
void cpu_fix_neuron_v1<float>(const int N, 
                               const float* src,
                               const float* fragpos, 
                               float* dst, 
                               int val_min,
                               int val_max, 
                               int keep_scale, 
                               int method);
template
void cpu_fix_neuron_v1<double>(const int N, 
                                const double* src,
                                const double* fragpos, 
                                double* dst, 
                                int val_min,
                                int val_max, 
                                int keep_scale, 
                                int method);
template<typename Dtype>
void cpu_fix_neuron_v2(const int N, 
                        const Dtype* src, 
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        Dtype val_amp, 
                        int zero_point,
                        int keep_scale, 
                        int method){
  _fix_neuron_v2(
    N, 
    src, 
    dst, 
    val_min,
    val_max, 
    val_amp, 
    zero_point,
    keep_scale,
    method);
}

template
void cpu_fix_neuron_v2<float>(const int N, 
                               const float* src,
                               float* dst, 
                               int val_min,
                               int val_max, 
                               float val_amp, 
                               int zero_point,
                               int keep_scale, 
                               int method);
template
void cpu_fix_neuron_v2<double>(const int N, 
                                const double* src,
                                double* dst, 
                                int val_min,
                                int val_max, 
                                double val_amp, 
                                int zero_point,
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
  if (step <= FLT_MIN) {
    max_scale = 18;
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
                      -(1<<(bitwidth-1)),
                      (1<<(bitwidth-1))-1,
                      Dtype(pow(2, scale)),
                      0,
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

static inline int float2int_cpu(const float x){
  return *(int*)&x; 
}

static inline float int2float_cpu(const int x){
  return *((float *)&x);
}

// int32 bitwise to bfloat16 with 32 bitwidth
static inline float int2bfloat_cpu(const int x){
  int itmp = x; // int32
  if ((itmp&0x00008000) == 0x00008000) { // half even
    if ((itmp&0xFFFF) > 0x00008000 || (((itmp&0xFFFF) == 0x00008000) && (itmp&0x10000) == 0x10000)){
      itmp += 0x10000;
    }
  }
  itmp &= 0xFFFF0000;
  return *(float *)&itmp; // int32 bitwise to float32
}

// float32 bitwise to bfloat16 with 32 bitwidth
static inline float float2bfloat_cpu(const float x){
  int itmp = *(int*)&x; // float32 bitwise to int32
  if ((itmp&0x00008000) == 0x00008000) { // half even
    if ((itmp&0xFFFF) > 0x00008000 || (((itmp&0xFFFF) == 0x00008000) && (itmp&0x10000) == 0x10000)){
      itmp += 0x10000;
    }
  }
  itmp &= 0xFFFF0000;
  return *(float *)&itmp; // int32 bitwise to float32
}

static inline int float2short_cpu(const float x){
  float itmp = float2bfloat_cpu(x);
  return *(int*)&itmp; 
}

// downshift one bit for short x
static inline int short_downshift_onebit_cpu(const int x){ // input: fake short
  int y, a, b;
  y = x >> 17;
  a = (x >> 16) & 1;
  if (y&1 == 1) // half2even
    y += a;
  return y << 17; // fake short
}

// aie sqrt(x) = x*(1/sqrt(x)) 
// with Newton iteration for 1/sqrt(x) 
void _sqrt(const float& x, float& out){
  float x2, y, y3h; 
  int i;
  x2 = x*0.5;
  y = x;
  i = float2int_cpu(y); // bitwise float32 to int32 
  i = (0x5f37 - (i >> 17)) << 16; // int32 -> int16 -> int32
  y = int2float_cpu(i); // initial value: bitwise int32 to float32

  // one step Newton iteration: y = y*(1.5 - (x2*y*y)) = 1.5*y - x2*y*y*y
  y3h = 1.5*y;  // float32
  y3h = float2bfloat_cpu(y3h); // bfloat with 32 bitwidth

  out = y*x2;  // float32
  out = float2bfloat_cpu(out); // bfloat with 32 bitwidth

  out = out*y;  // float32
  out = float2bfloat_cpu(out); // bfloat with 32 bitwidth

  out = out*y;
  out = float2bfloat_cpu(out); // bfloat with 32 bitwidth
  
  out = y3h - out; // float32: 1/sqrt(x) = 1.5*y - x2*y*y*y
  out = float2bfloat_cpu(out); // bfloat with 32 bitwidth

  out = x*out; // sqrt(x) = x*(1/sqrt(x))
  out = float2bfloat_cpu(out); // bfloat with 32 bitwidth
}

template<typename Dtype>
static void _aie_sqrt(const int N, 
                      const Dtype* src, 
                      Dtype* dst){
  for(int index=0; index < N; index++){
    float result_ = 1.0;
    _sqrt(src[index], result_); 
    dst[index] = result_; // auto cast: float to Dtype
  }
}

template<typename Dtype>
void cpu_aie_sqrt(const int N, const Dtype* src, Dtype* dst){ 
  _aie_sqrt(N, src, dst);
}

template
void cpu_aie_sqrt<float>(const int N, const float* src, float* dst); 

template
void cpu_aie_sqrt<double>(const int N, const double* src, double* dst);

// aie isqrt bfloat16 iteration
// 1/sqrt(x)
void _isqrt(const float& x, float& y){
  float x2, y2, mul2, mul, sub, threehalfs; 
  int i;
  x2  = x*0.5;
  x2 = float2bfloat_cpu(x2); // bitwise float32 to bfloat16

  i = float2short_cpu(x); // bitwise float32 to short (int16) 
  i = (0x5f37 - (short_downshift_onebit_cpu(i) >> 17)) << 16;

  y = int2bfloat_cpu(i); // initial value: bitwise int32 to bfloat16
  threehalfs = float2bfloat_cpu(1.5);

  // 4-steps-Newton iteration: y = y*(1.5 - (x2*y*y))
  for(int i=0; i<4; i++){
    y2 = y*y;
    y2 = float2bfloat_cpu(y2); // bfloat with 32 bitwidth

    mul2 = x2*y2;
    mul2 = float2bfloat_cpu(mul2);

    sub = threehalfs - mul2;
    sub = float2bfloat_cpu(sub);

    mul = y*sub;
    y = float2bfloat_cpu(mul);
  }
}

template<typename Dtype>
static void _aie_isqrt(const int N, 
                       const Dtype* src, 
                       Dtype* dst){
  for(int index=0; index < N; index++){
    float result_ = 1.0;
    _isqrt(src[index], result_); 
    dst[index] = result_; // auto cast: float to Dtype
  }
}

template<typename Dtype>
void cpu_aie_isqrt(const int N, const Dtype* src, Dtype* dst){ 
  _aie_isqrt(N, src, dst);
}

template
void cpu_aie_isqrt<float>(const int N, const float* src, float* dst); 

template
void cpu_aie_isqrt<double>(const int N, const double* src, double* dst);

// Layernorm isqrt float32 iteration
// 1/sqrt(x)
void isqrt(const float& x, float& y){
  float x2, threehalfs; 
  int i;
  x2 = x*0.5;
  y = x;
  threehalfs = 1.5;
  i = float2int_cpu(y); // bitwise float32 to int32 
  i = 0x5f3759df - (i >> 1);
  y = int2float_cpu(i); // bitwise int32 to float32
  y = y*(threehalfs - (x2*y*y)); // Newton steps
  y = y*(threehalfs - (x2*y*y));
  y = y*(threehalfs - (x2*y*y));
  y = y*(threehalfs - (x2*y*y));
}

template<typename Dtype>
static void _layernorm_isqrt(const int N, 
                             const Dtype* src, 
                             Dtype* dst){
  for(int index=0; index < N; index++){
    float result_ = 1.0;
    isqrt(src[index], result_); 
    dst[index] = result_; // auto cast: float to Dtype
  }
}

template<typename Dtype>
void cpu_layernorm_isqrt(const int N, const Dtype* src, Dtype* dst){ 
  _layernorm_isqrt(N, src, dst);
}

template
void cpu_layernorm_isqrt<float>(const int N, const float* src, float* dst); 

template
void cpu_layernorm_isqrt<double>(const int N, const double* src, double* dst);

