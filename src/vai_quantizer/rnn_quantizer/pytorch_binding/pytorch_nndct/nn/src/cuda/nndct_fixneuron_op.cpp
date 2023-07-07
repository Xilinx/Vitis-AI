

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
#include "../../include/nndct_fixneuron_op.h"
#include "../../../../../include/cuda/nndct_fix_kernels.h"
#include "../../../../../include/cpu/nndct_fix_kernels_cpu.h"

template <typename Dtype>
void _FixNeuronV2(Tensor Tinput,
                 Tensor Toutput,
                 int valmax,
                 Dtype valamp,
                 int method,
                 int device_id){

    auto input = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    if(device_id == 0){
      cuda_fix_neuron_v2(num_ele,
                         input,
                         output,
                         valmax,
                         valamp,
                         1, //keep_scale
                         method);
    }
    else if(device_id == 1){
      cpu_fix_neuron_v2(num_ele,
                        input,
                        output,
                        valmax,
                        valamp,
                        1, //keep_scale
                        method);
    }
}

void FixNeuronV2(Tensor Tinput,
                 Tensor Toutput,
                 int valmax,
                 float valamp,
                 int method,
                 int device_id){
  if (Tinput.dtype() == at::kFloat) {
    _FixNeuronV2<float>(Tinput,
                        Toutput,
                        valmax,
                        valamp,
                        method,
                        device_id);
  } else if (Tinput.dtype() == at::kDouble) {
    _FixNeuronV2<double>(Tinput,
                         Toutput,
                         valmax,
                         valamp,
                         method,
                         device_id);
  } else {
    LOG(FATAL) << "Unsupported tensor type: " << Tinput.toString();
  }
}

