

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
#include "../include/nndct_math_gpu.h"
#include  "../../../../include/nndct_cuda_math.h"
#include  "../../../../include/nndct_fix_kernels.h"

template <typename Dtype>
void _Scale(Tensor Tinput, Dtype scale) 
{
    auto input = Tinput.data<Dtype>();

    int64_t num_ele = Tinput.numel();
    cuda_scale_inplace(num_ele, input, scale);
}

void Scale(Tensor Tinput, float scale) {
  if (Tinput.dtype() == at::kFloat)
    _Scale<float>(Tinput, scale);
  else if (Tinput.dtype() == at::kDouble)
    _Scale<double>(Tinput, scale);
}

template <typename Dtype>
void _SigmoidTableLookup(Tensor Tinput, 
                        Tensor Ttable, 
                        Tensor Toutput, 
                        int fragpos)
{
    auto input  = Tinput.data<Dtype>();
    auto table  = Ttable.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    cuda_sigmoid_table_lookup(num_ele, input, table, output, fragpos);
}

void SigmoidTableLookup(Tensor Tinput, 
                        Tensor Ttable, 
                        Tensor Toutput, 
                        int fragpos){
  if (Tinput.dtype() == at::kFloat)
    _SigmoidTableLookup<float>(Tinput, 
                               Ttable, 
                               Toutput, 
                               fragpos);
  else if (Tinput.dtype() == at::kDouble)
    _SigmoidTableLookup<double>(Tinput, 
                                Ttable, 
                                Toutput, 
                                fragpos);
}

template <typename Dtype>
void _TanhTableLookup(Tensor Tinput, 
                     Tensor Ttable, 
                     Tensor Toutput, 
                     int fragpos)
{
    auto input  = Tinput.data<Dtype>();
    auto table  = Ttable.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    cuda_tanh_table_lookup(num_ele, input, table, output, fragpos);
}

void TanhTableLookup(Tensor Tinput, 
                     Tensor Ttable, 
                     Tensor Toutput, 
                     int fragpos) {
  if (Tinput.dtype() == at::kFloat)
    _TanhTableLookup<float>(Tinput, 
                            Ttable, 
                            Toutput, 
                            fragpos);
  else if (Tinput.dtype() == at::kDouble)
    _TanhTableLookup<double>(Tinput, 
                             Ttable, 
                             Toutput, 
                             fragpos);
}
