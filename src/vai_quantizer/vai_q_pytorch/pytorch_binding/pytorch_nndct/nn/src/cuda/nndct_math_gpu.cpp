

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

#include <ATen/ATen.h>
#include "c10/util/ArrayRef.h"
#include "../../include/nndct_math.h"
#include  "../../../../../include/cuda/nndct_cuda_math.h"
#include  "../../../../../include/cpu/nndct_cpu_math.h"
#include  "../../../../../include/cuda/nndct_fix_kernels.h"
#include  "../../../../../include/cpu/nndct_fix_kernels_cpu.h"

template <typename Dtype>
void _Scale(Tensor Tinput, Dtype scale, int device_id) 
{
    auto input = Tinput.data<Dtype>();
    
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_scale_inplace(num_ele, input, scale);
    else if(device_id == 1)
      cpu_scale_inplace(num_ele, input,  scale);
}

void Scale(Tensor Tinput, float scale, int device_id) {
  if (Tinput.dtype() == at::kFloat)
    _Scale<float>(Tinput, scale, device_id);
  else if (Tinput.dtype() == at::kDouble)
    _Scale<double>(Tinput, scale, device_id);
}

template <typename Dtype>
void _SigmoidTableLookup(Tensor Tinput, 
                         Tensor Ttable, 
                         Tensor Toutput, 
                         int fragpos,
                         int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto table  = Ttable.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_sigmoid_table_lookup(num_ele, input, table, output, fragpos);
    else if(device_id == 1)
      cpu_sigmoid_table_lookup(num_ele, input, table, output, fragpos);
}

void SigmoidTableLookup(Tensor Tinput, 
                        Tensor Ttable, 
                        Tensor Toutput, 
                        int64_t fragpos,
                        int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _SigmoidTableLookup<float>(Tinput, 
                               Ttable, 
                               Toutput, 
                               fragpos,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _SigmoidTableLookup<double>(Tinput, 
                                Ttable, 
                                Toutput, 
                                fragpos,
                                device_id);
}

template <typename Dtype>
void _SigmoidSimulation(Tensor Tinput, 
                         Tensor Toutput, 
                         int fragpos,
                         int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_sigmoid_simulation(num_ele, input, output, fragpos);
}

void SigmoidSimulation(Tensor Tinput, 
                        Tensor Toutput, 
                        int64_t fragpos,
                        int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _SigmoidSimulation<float>(Tinput, 
                               Toutput, 
                               fragpos,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _SigmoidSimulation<double>(Tinput, 
                                Toutput, 
                                fragpos,
                                device_id);
}

template <typename Dtype>
void _TanhTableLookup(Tensor Tinput, 
                      Tensor Ttable, 
                      Tensor Toutput, 
                      int fragpos,
                      int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto table  = Ttable.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_tanh_table_lookup(num_ele, input, table, output, fragpos);
    else if(device_id == 1)
      cpu_tanh_table_lookup(num_ele, input, table, output, fragpos);
}

void TanhTableLookup(Tensor Tinput, 
                     Tensor Ttable, 
                     Tensor Toutput, 
                     int64_t fragpos,
                     int64_t device_id) {
  if (Tinput.dtype() == at::kFloat)
    _TanhTableLookup<float>(Tinput, 
                            Ttable, 
                            Toutput, 
                            fragpos,
                            device_id);
  else if (Tinput.dtype() == at::kDouble)
    _TanhTableLookup<double>(Tinput, 
                             Ttable, 
                             Toutput, 
                             fragpos,
                             device_id);
}

template <typename Dtype>
void _TanhSimulation(Tensor Tinput, 
                      Tensor Toutput, 
                      int fragpos,
                      int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_tanh_simulation(num_ele, input, output, fragpos);
}

void TanhSimulation(Tensor Tinput, 
                     Tensor Toutput, 
                     int64_t fragpos,
                     int64_t device_id) {
  if (Tinput.dtype() == at::kFloat)
    _TanhSimulation<float>(Tinput, 
                            Toutput, 
                            fragpos,
                            device_id);
  else if (Tinput.dtype() == at::kDouble)
    _TanhSimulation<double>(Tinput, 
                             Toutput, 
                             fragpos,
                             device_id);
}

template <typename Dtype>
void _SoftmaxExpApproximate(Tensor Tinput, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if(device_id == 0)
      cuda_softmax_exp_approximate(num_ele, input, output);
}

void SoftmaxExpApproximate(Tensor Tinput, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _SoftmaxExpApproximate<float>(Tinput, 
                               Toutput,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _SoftmaxExpApproximate<double>(Tinput, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _SoftmaxLOD(Tensor Tinput, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if(device_id == 0)
      cuda_softmax_lod(num_ele, input, output);
}

void SoftmaxLOD(Tensor Tinput, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _SoftmaxLOD<float>(Tinput, 
                               Toutput,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _SoftmaxLOD<double>(Tinput, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _SoftmaxSimulationPart1(Tensor Tinput, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if(device_id == 0)
      cuda_softmax_simulation_part_1(num_ele, input, output);
}

void SoftmaxSimulationPart1(Tensor Tinput, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _SoftmaxSimulationPart1<float>(Tinput, 
                               Toutput,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _SoftmaxSimulationPart1<double>(Tinput, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _SoftmaxSimulationPart2(Tensor Sum, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto output  = Toutput.data<Dtype>();
    auto sum  = Sum.data<Dtype>();
    int64_t num_ele = Toutput.numel();
    
    if(device_id == 0)
      cuda_softmax_simulation_part_2(num_ele, sum, output);
}

void SoftmaxSimulationPart2(Tensor Sum, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Toutput.dtype() == at::kFloat)
    _SoftmaxSimulationPart2<float>(Sum, 
                               Toutput,
                               device_id);
  else if (Toutput.dtype() == at::kDouble)
    _SoftmaxSimulationPart2<double>(Sum, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _SigmoidTableLookupAIE2(Tensor Tinput, 
                         Tensor Toutput, 
                         int fragpos,
                         int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_sigmoid_table_lookup_aie2(num_ele, input, output, fragpos);
}

void SigmoidTableLookupAIE2(Tensor Tinput, 
                        Tensor Toutput, 
                        int64_t fragpos,
                        int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _SigmoidTableLookupAIE2<float>(Tinput, 
                               Toutput, 
                               fragpos,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _SigmoidTableLookupAIE2<double>(Tinput, 
                                Toutput, 
                                fragpos,
                                device_id);
}

template <typename Dtype>
void _TanhTableLookupAIE2(Tensor Tinput, 
                      Tensor Toutput, 
                      int fragpos,
                      int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_tanh_table_lookup_aie2(num_ele, input, output, fragpos);
}

void TanhTableLookupAIE2(Tensor Tinput, 
                     Tensor Toutput, 
                     int64_t fragpos,
                     int64_t device_id) {
  if (Tinput.dtype() == at::kFloat)
    _TanhTableLookupAIE2<float>(Tinput, 
                            Toutput, 
                            fragpos, 
                            device_id);
  else if (Tinput.dtype() == at::kDouble)
    _TanhTableLookupAIE2<double>(Tinput, 
                             Toutput, 
                             fragpos,
                             device_id);
}

template <typename Dtype>
void _ExpApprAIE2(Tensor Tinput, 
                      Tensor Toutput, 
                      int64_t bit_width,
                      int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_exp_appr_aie2(num_ele, input, output, bit_width);
}

void ExpApprAIE2(Tensor Tinput, 
                     Tensor Toutput, 
                     int64_t bit_width,
                     int64_t device_id) {
  if (Tinput.dtype() == at::kFloat)
    _ExpApprAIE2<float>(Tinput, 
                            Toutput, 
                            bit_width,
                            device_id);
  else if (Tinput.dtype() == at::kDouble)
    _ExpApprAIE2<double>(Tinput, 
                             Toutput, 
                             bit_width,
                             device_id);
}

template <typename Dtype>
void _LogSoftmaxFastLn(Tensor Tinput, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if(device_id == 0)
      cuda_log_softmax_fast_ln(num_ele, input, output);
}

void LogSoftmaxFastLn(Tensor Tinput, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _LogSoftmaxFastLn<float>(Tinput, 
                               Toutput,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _LogSoftmaxFastLn<double>(Tinput, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _LogSoftmaxSub(Tensor Tinput, 
                         Tensor Toutput,
                         Tensor Tsum, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    auto sum  = Tsum.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if(device_id == 0)
      cuda_log_softmax_sub(num_ele, input, output, sum);
}

void LogSoftmaxSub(Tensor Tinput, 
                         Tensor Toutput,
                         Tensor Tsum, 
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _LogSoftmaxSub<float>(Tinput, 
                               Toutput,
                               Tsum,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _LogSoftmaxSub<double>(Tinput, 
                                Toutput,
                                Tsum,
                                device_id);
}

template <typename Dtype>
void _LayernormISqrt(Tensor Tinput, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if (device_id == 0)
      cuda_layernorm_isqrt(num_ele, input, output);
    else if (device_id == 1)
      cpu_layernorm_isqrt(num_ele, input, output);
}

void LayernormISqrt(Tensor Tinput, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _LayernormISqrt<float>(Tinput, 
                               Toutput,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _LayernormISqrt<double>(Tinput, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _LayernormInvSqrt(Tensor Tinput, 
                         Tensor Toutput, 
                         int64_t device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output  = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    
    if(device_id == 0)
      cuda_layernorm_invsqrt(num_ele, input, output);
}

void LayernormInvSqrt(Tensor Tinput, 
                         Tensor Toutput,
                         int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _LayernormInvSqrt<float>(Tinput, 
                               Toutput,
                               device_id);
  else if (Tinput.dtype() == at::kDouble)
    _LayernormInvSqrt<double>(Tinput, 
                                Toutput,
                                device_id);
}

template <typename Dtype>
void _InverseAIE2(Tensor Tinput, 
                      Tensor Toutput, 
                      int device_id)
{
    auto input  = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0)
      cuda_inverse_aie2(num_ele, input, output);
}

void InverseAIE2(Tensor Tinput, 
                     Tensor Toutput, 
                     int64_t device_id) {
  if (Tinput.dtype() == at::kFloat)
    _InverseAIE2<float>(Tinput, 
                            Toutput, 
                            device_id);
  else if (Tinput.dtype() == at::kDouble)
    _InverseAIE2<double>(Tinput, 
                             Toutput, 
                             device_id);
}
