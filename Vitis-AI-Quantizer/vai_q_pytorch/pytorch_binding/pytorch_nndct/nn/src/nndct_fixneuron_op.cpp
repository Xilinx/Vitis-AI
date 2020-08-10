

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
#include "../include/nndct_fixneuron_op.h"
#include  "../../../../include/nndct_fix_kernels.h"

template <typename Dtype>
void _FixNeuronV2(Tensor Tinput,
                 Tensor Toutput, 
                 int valmax, 
                 Dtype valamp, 
                 int method){

    auto input = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    cuda_fix_neuron_v2(num_ele, 
                       input, 
                       output,
                       valmax, 
                       valamp, 
                       1, //keep_scale
                       method);
}

void FixNeuronV2(Tensor Tinput,
                 Tensor Toutput, 
                 int valmax, 
                 float valamp, 
                 int method){
  if (Tinput.dtype() == at::kFloat)
    _FixNeuronV2<float>(Tinput,
                        Toutput, 
                        valmax, 
                        valamp, 
                        method);
  else if (Tinput.dtype() == at::kDouble)
    _FixNeuronV2<double>(Tinput,
                         Toutput, 
                         valmax, 
                         valamp, 
                         method);
}

