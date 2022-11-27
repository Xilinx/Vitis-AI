

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

template<typename Dtype>
__global__ static void _fix_neuron_v1(const int N, 
                                      const Dtype* src, 
                                      const Dtype* fragpos, 
                                      Dtype* dst, 
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
__global__ static void _fix_neuron_v2(const int N, 
                                      const Dtype* src, 
                                      Dtype* dst, 
                                      int val_max, 
                                      Dtype val_amp, 
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
void cuda_fix_neuron_v1(const int N, 
                        const Dtype* src, 
                        const Dtype* fragpos, 
                        Dtype* dst, 
                        int val_max, 
                        int keep_scale, 
                        int method){
  _fix_neuron_v1<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
    N, 
    src, 
    fragpos, 
    dst, 
    val_max, 
    keep_scale,
    method);
}

template
void cuda_fix_neuron_v1<float>(const int N, 
                               const float* src,
                               const float* fragpos, 
                               float* dst, 
                               int val_max, 
                               int keep_scale, 
                               int method);
template
void cuda_fix_neuron_v1<double>(const int N, 
                                const double* src,
                                const double* fragpos, 
                                double* dst, 
                                int val_max, 
                                int keep_scale, 
                                int method);
template<typename Dtype>
void cuda_fix_neuron_v2(const int N, 
                        const Dtype* src, 
                        Dtype* dst, 
                        int val_max, 
                        Dtype val_amp, 
                        int keep_scale, 
                        int method){
  _fix_neuron_v2<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
    N, 
    src, 
    dst, 
    val_max, 
    val_amp, 
    keep_scale,
    method);
}

template
void cuda_fix_neuron_v2<float>(const int N, 
                               const float* src,
                               float* dst, 
                               int val_max, 
                               float val_amp, 
                               int keep_scale, 
                               int method);
template
void cuda_fix_neuron_v2<double>(const int N, 
                                const double* src,
                                double* dst, 
                                int val_max, 
                                double val_amp, 
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
  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0) {
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
                       1<<(bitwidth-1),
                       Dtype(pow(2, scale)),
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

template<typename Dtype>
__global__ static void _sigmoid_simulation(const int N,
                                             const Dtype fuzz,
                                             const Dtype* input,
                                             Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    if (input[i] >= 8.0) 
      output[i] = 1.0 - fuzz;
    else if (input[i] < -8.0)
      output[i] = 0.0;
    else {
      int x = int(input[i] * pow(2, 9));
      output[i] = sigmoid_short_sim(x, 9, 8) / (pow(2.0, 8));
    }
  }
}

template<typename Dtype>
void cuda_sigmoid_simulation(const int N, 
                               const Dtype* input, 
                               Dtype* output)
{
  Dtype fuzz = 1.0 / 32768;
  _sigmoid_simulation<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fuzz, 
      input, 
      output);
} 

template
void cuda_sigmoid_simulation<float>(const int N, 
                                      const float* input, 
                                      float* output);
template
void cuda_sigmoid_simulation<double>(const int N, 
                                       const double* input, 
                                       double* output);

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
                                          const Dtype fuzz,
                                          const Dtype* input,
                                          Dtype* output) {
  NNDCT_KERNEL_LOOP(i, N){
    if (input[i] >= 4.0) 
      output[i] = 1.0 - fuzz;
    else if (input[i] < -4.0)
      output[i] = -1.0;
    else {
      int x = int(input[i] * pow(2, 9));
      output[i] = tanh_short_sim(x, 9, 8) / (pow(2.0, 8));
    }
  }
}

template<typename Dtype>
void cuda_tanh_simulation(const int N, 
                            const Dtype* input, 
                            Dtype* output)
{
  Dtype fuzz = 1.0 / 32768;
  _tanh_simulation<<<NNDCT_GET_BLOCKS(N),NNDCT_CUDA_NUM_THREADS>>>(
      N, 
      fuzz, 
      input, 
      output);
} 

template
void cuda_tanh_simulation<float>(const int N, 
                                   const float* input, 
                                   float* output);
template
void cuda_tanh_simulation<double>(const int N, 
                                    const double* input, 
                                    double* output);

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
