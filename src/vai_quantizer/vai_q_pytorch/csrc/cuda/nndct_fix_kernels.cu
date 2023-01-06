

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
#include "../../include/cuda/nndct_fix_kernels.cuh"
#include "../../include/cuda/nndct_fix_kernels.h"
#include "../../include/cuda/nndct_cuda_math.h"
#include "../../include/cuda/nndct_cu_utils.h"
#include "../../include/cuda/table_data.h"

template<typename Dtype>
__global__ static void _fix_neuron_v1(const int N, 
                                      const Dtype* src, 
                                      const Dtype* fragpos, 
                                      Dtype* dst, 
                                      int val_min,
                                      int val_max, 
                                      int keep_scale, 
                                      int method){
  NNDCT_KERNEL_LOOP(index, N){
    //method:
    //1: dummy
    //2: for CNN feature map
    //3: for weights and bias
    //4: for RNN feature map
    int result_ = 0;
    Dtype val_amp = pow(2, *fragpos);
    _fix_neuron_v2_device(src[index], 
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
__global__ static void _fix_neuron_v2(const int N, 
                                      const Dtype* src, 
                                      Dtype* dst, 
                                      int val_min,
                                      int val_max, 
                                      Dtype val_amp, 
                                      int zero_point,
                                      int keep_scale, 
                                      int method){
  NNDCT_KERNEL_LOOP(index, N){
    //method: 
    //1: dummy
    //2: for CNN feature map
    //3: for weights and bias
    //4: for RNN feature map
    int result_ = 0;
    _fix_neuron_v2_device(src[index], 
                          result_, 
                          val_min,
                          val_max, 
                          val_amp, 
                          zero_point,
                          method);
    //printf( "$$$$$$$$$$$ result: %d zero_point: %d keep_scale: %d\n", result_, zero_point, keep_scale); 
    if(0 != keep_scale)
      dst[index] = (Dtype(result_)-Dtype(zero_point)) * (1 / val_amp);
    else
      dst[index] = result_;
  }
}

template<typename Dtype>
void cuda_fix_neuron_v1(const int N, 
                        const Dtype* src, 
                        const Dtype* fragpos, 
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        int keep_scale, 
                        int method){
  _fix_neuron_v1<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
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
void cuda_fix_neuron_v1<float>(const int N, 
                               const float* src,
                               const float* fragpos, 
                               float* dst, 
                               int val_min,
                               int val_max, 
                               int keep_scale, 
                               int method);
template
void cuda_fix_neuron_v1<double>(const int N, 
                                const double* src,
                                const double* fragpos, 
                                double* dst, 
                                int val_min,
                                int val_max, 
                                int keep_scale, 
                                int method);
template<typename Dtype>
void cuda_fix_neuron_v2(const int N, 
                        const Dtype* src, 
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        Dtype val_amp, 
                        int zero_point,
                        int keep_scale, 
                        int method){
  _fix_neuron_v2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
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
void cuda_fix_neuron_v2<float>(const int N, 
                               const float* src,
                               float* dst, 
                               int val_min,
                               int val_max, 
                               float val_amp, 
                               int zero_point,
                               int keep_scale, 
                               int method);
template
void cuda_fix_neuron_v2<double>(const int N, 
                                const double* src,
                                double* dst, 
                                int val_min,
                                int val_max, 
                                double val_amp, 
                                int zero_point,
                                int keep_scale, 
                                int method);

template<typename Dtype> 
void cuda_diff_S(const int N, 
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
  cuda_max(N, src, buffer);
  cudaMemcpy(&x_max, buffer, sizeof(Dtype), cudaMemcpyDeviceToHost);
  cuda_min(N, src, buffer);
  cudaMemcpy(&x_min, buffer, sizeof(Dtype), cudaMemcpyDeviceToHost);
  
  // Find max_scale
  // Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  // Hipify thinks std::max is kernel code so converts it to ::max
  // which doesn't behave correctly on the host side
  Dtype step = x_min / fix_lb;
  Dtype maxs = x_max / fix_ub;
  if (maxs > step) step = maxs;
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
    cuda_fix_neuron_v2(N,
                       src,
                       buffer,
                       -(1<<(bitwidth-1)),
                       (1<<(bitwidth-1))-1,
                       Dtype(pow(2, scale)),
                       0,
                       1,
                       method);
    cuda_sub(N, src, buffer);
    cuda_pow(N, buffer, Dtype(2));
    Dtype fixed_diff;
    cuda_sum_inplace(N, buffer);
    cudaMemcpy(&fixed_diff, 
               buffer, 
               sizeof(Dtype), 
               cudaMemcpyDeviceToHost);
    if (fixed_diff < fixed_diff_min) {
      final_scale = scale;
      fixed_diff_min = fixed_diff;
    }
  }
  //final_scale = final_scale > 15 ? 15: final_scale;
  cuda_set(1, output, final_scale);
#if 0
  printf( "$$$$$$$$$$$ diffs scale is %g, setting to %p...\n", 
          final_scale,
          output ); fflush(stdout);
#endif
}

template 
void cuda_diff_S<float>(const int N, 
                        const float* src, 
                        float* buffer, 
                        float* output, 
                        int bitwidth, 
                        int range, 
                        int method);
template 
void cuda_diff_S<double>(const int N, 
                         const double* src, 
                         double* buffer, 
                         double* output, 
                         int bitwidth, 
                         int range, 
                         int method);

/*
Sigmoid & Tanh table look up FPGA 
*/

template<typename Dtype>
__global__ static void _sigmoid_table_lookup(const int N,
                                             const int fragpos,
                                             const Dtype scale,
                                             const Dtype fuzz,
                                             const Dtype* input,
                                             const Dtype* table,
                                             Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
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
void cuda_sigmoid_table_lookup(const int N, 
                               const Dtype* input, 
                               const Dtype* table,
                               Dtype* output,
                               int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _sigmoid_table_lookup<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fragpos, 
      scale, 
      fuzz, 
      input, 
      table, 
      output);
} 

template
void cuda_sigmoid_table_lookup<float>(const int N, 
                                      const float* input, 
                                      const float* table,
                                      float* output,
                                      int fragpos);
template
void cuda_sigmoid_table_lookup<double>(const int N, 
                                       const double* input, 
                                       const double* table,
                                       double* output,
                                       int fragpos);

template<typename Dtype>
__global__ static void _tanh_table_lookup(const int N,
                                          const int fragpos,
                                          const Dtype scale,
                                          const Dtype fuzz,
                                          const Dtype* input,
                                          const Dtype* table,
                                          Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
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
void cuda_tanh_table_lookup(const int N, 
                            const Dtype* input, 
                            const Dtype* table,
                            Dtype* output,
                            int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _tanh_table_lookup<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fragpos, 
      scale, 
      fuzz, 
      input, 
      table, 
      output);
} 

template
void cuda_tanh_table_lookup<float>(const int N, 
                                   const float* input, 
                                   const float* table,
                                   float* output,
                                   int fragpos);
template
void cuda_tanh_table_lookup<double>(const int N, 
                                    const double* input, 
                                    const double* table,
                                    double* output,
                                    int fragpos);

/*
Sigmoid & Tanh simulation 
*/

#define C_P0       1.98364257812E-4
#define CF_P1      1.3981999507E-3 
#define CF_P2      8.3334519073E-3 
#define CF_P3      4.1665795894E-2 
#define CF_P4      1.6666665459E-1 
#define CF_P5      5.0000001201E-1 

__device__ static inline short bfloat16(float x){
 int itmp =  *(int*)&x;
  if((itmp&0x00008000) == 0x00008000)
    itmp += 0x00010000;
  return (short)((itmp>>16)&0xFFFF);
}

__device__ static inline float rbfloat(short x){
  int itmp = (x<<16)&0xFFFF0000;
  return *((float *)&itmp);
}

__device__ static inline float as_bfloat16(float x){
  int itmp =  *(int*)&x;
  if((itmp&0x00008000) == 0x00008000)
    itmp += 0x00010000;
  itmp &= 0xFFFF0000;
  return *(float *)&itmp;
}

__device__ static inline float as_bfloat16_numpy(float x){
  int itmp =  *(int*)&x;
  itmp &= 0xFFFF0000;
  return *(float *)&itmp;
}

__device__ static inline int float2int_cuda(float x){
  return *(int*)&x; 
}

__device__ static inline float int2float_cuda(int x){
  return *((float *)&x);
}

__device__ float exp_sim(short x)
{
  float ftmp, fz, fres, fx;

  ftmp = rbfloat(x);
  fres = ftmp*1.4375+0.5;
  //round
  fres += 12582912.0;
  fres -= 12582912.0;
  //round end
  fz = fres;
  fx = ftmp - fres*0.69140625;
  fres = as_bfloat16(fx)*C_P0 + CF_P1; 
  fres = as_bfloat16(fx)*as_bfloat16(fres) + CF_P2; 
  fres = as_bfloat16(fx)*as_bfloat16(fres) + CF_P3; 
  fres = as_bfloat16(fx)*as_bfloat16(fres) + CF_P4; 
  fres = as_bfloat16(fx)*as_bfloat16(fres) + CF_P5; 
  fres = as_bfloat16(fx)*as_bfloat16(fres) + 1.0; 
  fres = as_bfloat16(fx)*as_bfloat16(fres) + 1.0; 

  fres = as_bfloat16(fres)*as_bfloat16(pow(2, fz));

  return as_bfloat16(fres);
}

__device__ float inv_sim(float x)
{
  float a = x;
  int tt = 0x7F000000 - *(int*)&a;
  float r = *(float *)&tt;
  float m;
  
  a = as_bfloat16(a);

  for (int k=0; k<4; k++){
    m = 2.0 - as_bfloat16(a)*as_bfloat16(r);
    r = as_bfloat16(m)*as_bfloat16(r);
  }

  return r;
}

__device__ float sigmoid_sim(short x)
{
  float fres = exp_sim(x);
  float r = inv_sim(as_bfloat16(fres)+1.0);
  fres = as_bfloat16(fres)*as_bfloat16(r);
  return as_bfloat16(fres);
}

__device__ short sigmoid_short_sim(short x, int ishift, int oshift)
{

  float fx = (float)x;
  float iscale = pow(2, -ishift);
  float res = as_bfloat16(iscale)*as_bfloat16(fx);
  res = sigmoid_sim(bfloat16(res));
  float oscale = pow(2, oshift);
  res = as_bfloat16(oscale)*as_bfloat16(res);
  
  float fy = as_bfloat16(res) + 12582912.0;
  int y = *((int*)&fy);
  y -= 0x4B400000;

  return y&0xFFFF;
}

template<typename Dtype>
__global__ static void _sigmoid_simulation(const int N,
                                            const int fragpos,
                                             const Dtype scale,
                                             const Dtype fuzz,
                                             const Dtype* input,
                                             Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    if (input[i] >= 8.0) 
      output[i] = 1.0 - fuzz;
    else if (input[i] < -8.0)
      output[i] = 0.0;
    else {
      int x = int(input[i] * scale);
      output[i] = sigmoid_short_sim(x, fragpos, 15) * fuzz;
    }
  }
}

template<typename Dtype>
void cuda_sigmoid_simulation(const int N, 
                               const Dtype* input, 
                               Dtype* output,
                               int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _sigmoid_simulation<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fragpos,
      scale,
      fuzz, 
      input, 
      output);
} 

template
void cuda_sigmoid_simulation<float>(const int N, 
                                      const float* input, 
                                      float* output,
                                      int fragpos);
template
void cuda_sigmoid_simulation<double>(const int N, 
                                       const double* input, 
                                       double* output,
                                       int fragpos);

__device__ float tanh_sim(short x)
{
  float fres = exp_sim(x);
  fres = as_bfloat16(fres)*as_bfloat16(fres);
  float r = inv_sim(as_bfloat16(fres)+1.0);
  fres = as_bfloat16(fres)-1.0;
  fres = as_bfloat16(fres)*as_bfloat16(r);
  return as_bfloat16(fres);
}

__device__ short tanh_short_sim(short x, int ishift, int oshift)
{
  float fx = (float)x;
  float iscale = pow(2, -ishift);
  float res = as_bfloat16(iscale)*as_bfloat16(fx);
  res = tanh_sim(bfloat16(res));
  float oscale = pow(2, oshift);
  res = as_bfloat16(oscale)*as_bfloat16(res);
  float fy = as_bfloat16(res) + 12582912.0;
  int y = *((int*)&fy);
  y -= 0x4B400000;
  
  return y&0xFFFF;
}

template<typename Dtype>
__global__ static void _tanh_simulation(const int N,
                                          const int fragpos,
                                          const Dtype scale,
                                          const Dtype fuzz,
                                          const Dtype* input,
                                          Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    if (input[i] >= 4.0) 
      output[i] = 1.0 - fuzz;
    else if (input[i] < -4.0)
      output[i] = -1.0;
    else {
      int x = int(input[i] * scale);
      output[i] = tanh_short_sim(x, fragpos, 15) * fuzz;
    }
  }
}

template<typename Dtype>
void cuda_tanh_simulation(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _tanh_simulation<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fragpos, 
      scale,
      fuzz, 
      input, 
      output);
} 

template
void cuda_tanh_simulation<float>(const int N, 
                                   const float* input, 
                                   float* output,
                                   int fragpos);
template
void cuda_tanh_simulation<double>(const int N, 
                                    const double* input, 
                                    double* output,
                                    int fragpos);

template<typename Dtype>
__global__ static void _softmax_exp_approximate(const int N,
                                         const Dtype* input,
                                         Dtype* output) {
  
  NNDCT_KERNEL_LOOP(i, N){
    Dtype u;
    Dtype v;
    if (input[i] >= 0){
      u = floor(input[i]);
      v = input[i] - u;
    }
    else{
      u = ceil(input[i]);
      v = input[i] - u;
    }
    
    if (v <= -0.75) 
      output[i] = ((12409.0/pow(2.0, 15)) * v + 28747.0/pow(2.0, 15))/pow(2.0, -u);
    else if (v <= -0.5)
      output[i] = ((14759.0/pow(2.0, 15)) * v + 30497.0/pow(2.0, 15))/pow(2.0, -u);
    else if (v <= -0.25)
      output[i] = ((17551.0/pow(2.0, 15)) * v + 31880.0/pow(2.0, 15))/pow(2.0, -u);
    else {
      output[i] = ((20873.0/pow(2.0, 15)) * v + 32696.0/pow(2.0, 15))/pow(2.0, -u);
    }
  }
}

/*
Hardware-PL softmax 
*/

template<typename Dtype>
void cuda_softmax_exp_approximate(const int N,
                            const Dtype* input,
                            Dtype* output)
{
  _softmax_exp_approximate<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output);
} 

template
void cuda_softmax_exp_approximate<float>(const int N,
                                    const float* input,
                                    float* output);
template
void cuda_softmax_exp_approximate<double>(const int N,
                                    const double* input,
                                    double* output);

template<typename Dtype>
__global__ static void _softmax_lod(const int N,
                                    const Dtype* input,
                                    Dtype* output)
{
  NNDCT_KERNEL_LOOP(i, N){
    float lod_s = 0;
    float s_int = floor(input[i]);
    while(s_int >= 2){
      lod_s += 1;
      s_int /= 2;
    }
    output[i] = lod_s;
  }
}

template<typename Dtype>
void cuda_softmax_lod(const int N,
                            const Dtype* input,
                            Dtype* output)
{
  _softmax_lod<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output);
} 

template
void cuda_softmax_lod<float>(const int N,
                                    const float* input,
                                    float* output);
template
void cuda_softmax_lod<double>(const int N,
                                    const double* input,
                                    double* output);

/*
Liyi softmax 
*/

template<typename Dtype>
__global__ static void _softmax_simulation_part_1(const int N,
                                    const Dtype* input,
                                    Dtype* output)
{
  float temp;
  int itemp;
  short bf16hex;
  float fres;
  NNDCT_KERNEL_LOOP(i, N){
    if (input[i] <= -80) 
      temp = -80;
    else{
      temp = input[i];
    }
    itemp = *((int *)&temp);
    bf16hex = (itemp>>16) & 0xFFFF;
    fres = exp_sim(bf16hex);
    output[i] = fres;
  }
}

template<typename Dtype>
void cuda_softmax_simulation_part_1(const int N,
                            const Dtype* input,
                            Dtype* output)
{
  _softmax_simulation_part_1<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output);
} 

template
void cuda_softmax_simulation_part_1<float>(const int N,
                                    const float* input,
                                    float* output);
template
void cuda_softmax_simulation_part_1<double>(const int N,
                                    const double* input,
                                    double* output);

template<typename Dtype>
__global__ static void _softmax_simulation_part_2(const int N,
                                         const Dtype* sum,
                                         Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    float r = inv_sim(as_bfloat16(sum[i]));
    output[i] = as_bfloat16(r);
  }
}

template<typename Dtype>
void cuda_softmax_simulation_part_2(const int N,
                            const Dtype* sum,
                            Dtype* output)
{
  _softmax_simulation_part_2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      sum,
      output);
} 

template
void cuda_softmax_simulation_part_2<float>(const int N,
                                    const float* sum,
                                    float* output);
template
void cuda_softmax_simulation_part_2<double>(const int N,
                                    const double* sum,
                                    double* output);

/*
Sigmoid & Tanh & LogSoftmax Table Look up AIE2
*/
#define STEP 0.0009765625f //16/256/256
#define LN2 0.69314718056f

__device__ static inline float short_to_float(short x){
    int itmp = (x<<16)&0xFFFF0000;
    float f = *((float *)&itmp);
    return f;
}

__device__ static inline short clip_int16(float x){
    short res;
    if(x > 32767)
        res = 32767;
    else if(x < -32768)
        res = -32768;
    else
        res = short(x);
    return res;
}

__device__ float vector_inv(float x){
    short x_as_int16 = bfloat16(x);
    short num = 0x3F80;
    short res_as_int16 = 2*num - x_as_int16;
    return short_to_float(res_as_int16);
}

__device__ float compute_inv(float x){

    unsigned int *B_x;
    unsigned int exp_mask       = 0x7F800000;
    unsigned int mantissa_mask  = 0x007FFFFF;
    unsigned int mantissa_Q     = 0x00008000;
    unsigned char exponent, mantissa;
    unsigned int inv_exponent;
    unsigned short inv_x_val;
    float inv_x_float;

    B_x = (unsigned int*)&x;
    exponent = (*B_x & exp_mask) >> 23;
    mantissa = ((*B_x & mantissa_Q)==0) ? ((*B_x & mantissa_mask)>>16) : ((*B_x & mantissa_mask)>>16)+1;
    inv_exponent = 253-exponent;
    if(mantissa > 127)
        mantissa = 127;
    inv_x_val = (inv_exponent<<7) + m_inv_lut[mantissa];
    inv_x_float = short_to_float(inv_x_val);
    return inv_x_float;
}

__device__ float compute_exp(short x){
    unsigned short x_no_sign, h_8bit, l_8bit, h, l;
    float f1, f2, res;

    x_no_sign = (unsigned short)x;
    h_8bit = (x_no_sign>>8)&0X00FF;
    l_8bit = x_no_sign&0X00FF;
    //if(l_8bit>=0x00E0 && l_8bit <=0x00FF)
    //    h_8bit += 0x0001;

    h = h_8bit/8*16 + h_8bit%8;
    l = l_8bit/8*16 + l_8bit%8;
    f1 = short_to_float(s_ilut_ab[h]);
    f2 = short_to_float(s_flut_ab[l]);
    res = as_bfloat16(f1*f2);

    return res;
}

__device__ float compute_exp_soft(short x){
    unsigned short x_no_sign, h_8bit, l_8bit, h, l;
    float f1, f2, res;

    x_no_sign = (unsigned short)x;
    h_8bit = (x_no_sign>>8)&0X00FF;
    l_8bit = x_no_sign&0X00FF;
    //if(l_8bit>=0x00E0 && l_8bit <=0x00FF)
    //    h_8bit += 0x0001;

    h = h_8bit/8*16 + h_8bit%8;
    l = l_8bit/8*16 + l_8bit%8;
    f1 = short_to_float(s_ilut_cd[h]);
    f2 = short_to_float(s_flut_cd[l]);
    res = as_bfloat16(f1*f2);

    return res;
}

// cubic approximation of ln(x) in range [1, 2]:
__device__ static inline float small_ln(float x) {
    x -= 1.0f;
    return 0.6931471805599453f*x*(1.4201157697141027f + x*(-0.5747927782450741f + x*(0.15468105905881002f)));
    }

//for ln(x) with x>0
__device__ float fast_ln(float x){
    unsigned char exponent;
    float mantissa, ln_mantissa;
    float res_ln;

    int x_int = *(int *)&x;
    exponent = (x_int&0x7f800000) >> 23;
    x_int &= 0x007FFFFF;//mask away the exp
    x_int |= 0x3F800000;//set the exp to 127
    mantissa = *(float *)&x_int;
    ln_mantissa = small_ln(mantissa);
    res_ln = ln_mantissa+(exponent-127)*LN2;
    return as_bfloat16(res_ln);
}

template<typename Dtype>
__global__ static void _sigmoid_table_lookup_aie2(const int N,
                                             const Dtype fuzz,
                                             const Dtype* input,
                                             Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    short in;
    float exp_sigm;
    float denom_sigm, denom_inv_sigm;
    float sigmoid, in_float, res_float;

    in_float = input[i] * pow(2.0, 12);
    in = clip_int16(in_float);
    exp_sigm = compute_exp(in);
    denom_sigm = exp_sigm + 1.0;
    denom_inv_sigm = vector_inv(denom_sigm);
    sigmoid = as_bfloat16(exp_sigm * denom_inv_sigm);
    res_float = as_bfloat16(sigmoid * pow(2.0, 15));
    output[i] = clip_int16(res_float)* pow(2.0, -15);

  }
}

template<typename Dtype>
void cuda_sigmoid_table_lookup_aie2(const int N, 
                               const Dtype* input, 
                               Dtype* output,
                               int fragpos)
{
  Dtype fuzz = 1.0 / 32768;
  _sigmoid_table_lookup_aie2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fuzz, 
      input, 
      output);
} 

template
void cuda_sigmoid_table_lookup_aie2<float>(const int N, 
                                      const float* input, 
                                      float* output,
                                      int fragpos);
template
void cuda_sigmoid_table_lookup_aie2<double>(const int N, 
                                       const double* input, 
                                       double* output,
                                       int fragpos);



template<typename Dtype>
__global__ static void _tanh_table_lookup_aie2(const int N,
                                          const Dtype fuzz,
                                          const Dtype* input,
                                          Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    short in;
    float temp_tanh;
    float denom_inv_tanh;
    float tanh, in_float, res_float;
    short res;

    in_float = input[i] * pow(2.0, 12);
    in = clip_int16(in_float);
    temp_tanh = compute_exp(in);
    temp_tanh = temp_tanh * temp_tanh;//e^2x
    temp_tanh = as_bfloat16(temp_tanh);
    denom_inv_tanh = vector_inv(temp_tanh + 1.0);//1/(e^2x + 1)
    temp_tanh = as_bfloat16(temp_tanh) - 1.0;//e^2x-1
    tanh = as_bfloat16(temp_tanh) * denom_inv_tanh;
    res_float = as_bfloat16(tanh * pow(2.0, 15));
    res = clip_int16(res_float);
    output[i] = res* pow(2.0, -15);
  }
}

template<typename Dtype>
void cuda_tanh_table_lookup_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            int fragpos)
{
  Dtype scale = pow(2.0, fragpos);
  Dtype fuzz = 1.0 / 32768;
  _tanh_table_lookup_aie2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fuzz, 
      input, 
      output);
} 

template
void cuda_tanh_table_lookup_aie2<float>(const int N, 
                                   const float* input, 
                                   float* output,
                                   int fragpos);
template
void cuda_tanh_table_lookup_aie2<double>(const int N, 
                                    const double* input, 
                                    double* output,
                                    int fragpos);

template<typename Dtype>
__global__ static void _exp_appr_aie2(const int N,
                                          const Dtype* input,
                                          Dtype* output,
                                          const int bit_width) {
  NNDCT_KERNEL_LOOP(i, N){
    float input_f;
    if(input[i] < -63.0){
      input_f = -63;
    }
    else{
      input_f = input[i];
    }
    output[i] = compute_exp_soft(short(input_f * -1024));
  }
}

template<typename Dtype>
void cuda_exp_appr_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            const int bit_width)
{
  _exp_appr_aie2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      input, 
      output,
      bit_width);
} 

template
void cuda_exp_appr_aie2<float>(const int N, 
                                   const float* input, 
                                   float* output,
                                   const int bit_width);
template
void cuda_exp_appr_aie2<double>(const int N, 
                                    const double* input, 
                                    double* output,
                                    const int bit_width);

template<typename Dtype>
__global__ static void _log_softmax_fast_ln(const int N,
                                         const Dtype* input,
                                         Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    output[i] = fast_ln(input[i]);
  }
}

template<typename Dtype>
void cuda_log_softmax_fast_ln(const int N,
                            const Dtype* input,
                            Dtype* output)
{
  _log_softmax_fast_ln<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output);
  
} 

template
void cuda_log_softmax_fast_ln<float>(const int N,
                                    const float* input,
                                    float* output);
template
void cuda_log_softmax_fast_ln<double>(const int N,
                                    const double* input,
                                    double* output);

template<typename Dtype>
__global__ static void _log_softmax_sub(const int N,
                                         const Dtype* input,
                                         Dtype* output,
                                         const Dtype* sum) {
  NNDCT_KERNEL_LOOP(i, N){
    output[i] = as_bfloat16(as_bfloat16((float)input[i])-sum[0]);
  }
}

template<typename Dtype>
void cuda_log_softmax_sub(const int N,
                            const Dtype* input,
                            Dtype* output,
                            const Dtype* sum)
{
  _log_softmax_sub<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output,
      sum);
  
} 

template
void cuda_log_softmax_sub<float>(const int N,
                                    const float* input,
                                    float* output,
                                    const float* sum);
template
void cuda_log_softmax_sub<double>(const int N,
                                    const double* input,
                                    double* output,
                                    const double* sum);

/*
Layernorm isqrt AIE2, float32 iteration
*/
__device__ float isqrt(float x){
  float x1, x2, y, threehalfs; 
  int i;
  x2  = x*0.5;
  y = x;
  threehalfs = 1.5;
  i = float2int_cuda(y); // bitwise float32 to int32 
  i = 0x5f3759df - (i >> 1);
  y = int2float_cuda(i); // bitwise int32 to float32

  y = y*(threehalfs - (x2*y*y)); // Newton steps
  y = y*(threehalfs - (x2*y*y));
  y = y*(threehalfs - (x2*y*y));
  y = y*(threehalfs - (x2*y*y));
  return y;
}

template<typename Dtype>
__global__ static void _layernorm_isqrt(const int N,
                                         const Dtype* input,
                                         Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    output[i] = isqrt(input[i]);
  }
}

template<typename Dtype>
void cuda_layernorm_isqrt(const int N,
                            const Dtype* input,
                            Dtype* output)
{
  _layernorm_isqrt<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output);
  
} 

template
void cuda_layernorm_isqrt<float>(const int N,
                                    const float* input,
                                    float* output);
template
void cuda_layernorm_isqrt<double>(const int N,
                                    const double* input,
                                    double* output);

/*
Layernorm Inv Sqrt AIE2
*/
__device__ float invsqrt(float x){
  x = as_bfloat16_numpy(x);
  
  short i;
  float x1,x2;
  float y1,y2,y;
  x1 = as_bfloat16_numpy(x * 0.5);
  x2 = as_bfloat16_numpy(x);

  i  = bfloat16(x2);  
  i  = 0x5f37 - ( i >> 1 );
  x2  = rbfloat(i);
  y2 = as_bfloat16_numpy(x2 * 1.5);

  y1 = x1*x2*x2*x2;
  y1 = as_bfloat16_numpy(y1);
  y = y2-y1;
  y = as_bfloat16_numpy(y);
  return y;
}

template<typename Dtype>
__global__ static void _layernorm_invsqrt(const int N,
                                         const Dtype* input,
                                         Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    output[i] = invsqrt(input[i]);
  }
}

template<typename Dtype>
void cuda_layernorm_invsqrt(const int N,
                            const Dtype* input,
                            Dtype* output)
{
  _layernorm_invsqrt<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N,
      input,
      output);
  
} 

template
void cuda_layernorm_invsqrt<float>(const int N,
                                    const float* input,
                                    float* output);
template
void cuda_layernorm_invsqrt<double>(const int N,
                                    const double* input,
                                    double* output);

/*
AIE2 Softmax
*/
template<typename Dtype>
__global__ static void _inverse_aie2(const int N,
                                          const Dtype* input,
                                          Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    
    output[i] = compute_inv(input[i]);
  }
}

template<typename Dtype>
void cuda_inverse_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output)
{
  _inverse_aie2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      input, 
      output);
} 

template
void cuda_inverse_aie2<float>(const int N, 
                                   const float* input, 
                                   float* output);
template
void cuda_inverse_aie2<double>(const int N, 
                                    const double* input, 
                                    double* output);
