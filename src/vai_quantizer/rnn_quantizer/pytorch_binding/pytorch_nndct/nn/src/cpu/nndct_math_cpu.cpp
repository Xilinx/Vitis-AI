

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
#include  "../../../../../include/cpu/nndct_cpu_math.h"
#include  "../../../../../include/cpu/nndct_fix_kernels_cpu.h"

template <typename Dtype>
void _Scale(Tensor Tinput, Dtype scale, int device_id) 
{
    auto input = Tinput.data<Dtype>();

    int64_t num_ele = Tinput.numel();
    
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

    cpu_sigmoid_table_lookup(num_ele, input, table, output, fragpos);
}

void SigmoidTableLookup(Tensor Tinput, 
                        Tensor Ttable, 
                        Tensor Toutput, 
                        int fragpos,
                        int device_id){
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
    
    cpu_tanh_table_lookup(num_ele, input, table, output, fragpos);
}

void TanhTableLookup(Tensor Tinput, 
                     Tensor Ttable, 
                     Tensor Toutput, 
                     int fragpos,
                     int device_id) {
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