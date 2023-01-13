

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
#include "../../../../../include/cpu/nndct_fix_kernels_cpu.h"

template <typename Dtype>
void _FixNeuronV2(Tensor Tinput,
                 Tensor Toutput, 
                 int valmin,
                 int valmax, 
                 Dtype valamp, 
                 int zero_point,
                 int method,
                 int device_id){

    auto input = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    cpu_fix_neuron_v2(num_ele, 
                      input, 
                      output,
                      valmin,
                      valmax, 
                      valamp, 
                      zero_point,
                      1, //keep_scale
                      method);
}

void FixNeuronV2(Tensor Tinput,
                 Tensor Toutput, 
                 int valmin,
                 int valmax, 
                 float valamp, 
                 int zero_point,
                 int method,
                 int device_id){
  if (Tinput.dtype() == at::kFloat)
    _FixNeuronV2<float>(Tinput,
                        Toutput, 
                        valmin,
                        valmax, 
                        valamp, 
                        zero_point,
                        method,
                        device_id);
  else if (Tinput.dtype() == at::kDouble)
    _FixNeuronV2<double>(Tinput,
                         Toutput, 
                         valmin,
                         valmax, 
                         valamp, 
                         zero_point,
                         method,
                         device_id);
}

at::Tensor fix_neuron(at::Tensor Tinput, int64_t valmin, int64_t valmax, double valamp, int64_t zero_point, int64_t method, int64_t device_id, int64_t inplace){
  if (inplace != 0) {
    if (Tinput.dtype() == at::kFloat) {
      _FixNeuronV2<float>(Tinput,
                          Tinput,
                          valmin,
                          valmax,
                          valamp,
                          zero_point,
                          method,
                          device_id);
    } else if (Tinput.dtype() == at::kDouble) {
      _FixNeuronV2<double>(Tinput,
                           Tinput,
                           valmin,
                           valmax,
                           valamp,
                           zero_point,
                           method,
                           device_id);
    } else {
      LOG(FATAL) << "Unsupported tensor type: " << Tinput.toString();
    }
    return Tinput;
  } else {
    at::Tensor Toutput = at::native::empty_like(Tinput);
    if (Tinput.dtype() == at::kFloat) {
      _FixNeuronV2<float>(Tinput,
                          Toutput,
                          valmin,
                          valmax,
                          valamp,
                          zero_point,
                          method,
                          device_id);
    } else if (Tinput.dtype() == at::kDouble) {
      _FixNeuronV2<double>(Tinput,
                           Toutput,
                           valmin,
                           valmax,
                           valamp,
                           zero_point,
                           method,
                           device_id);
    } else {
      LOG(FATAL) << "Unsupported tensor type: " << Tinput.toString();
    }
    return Toutput;
  }
}

