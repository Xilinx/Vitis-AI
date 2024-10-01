

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
void _Round(Tensor Tinput, Tensor Toutput, int64_t method, int64_t device_id){

  auto input = Tinput.data<Dtype>();
  auto output = Toutput.data<Dtype>();
  int64_t num_ele = Tinput.numel();

  if(device_id == 0){
    cuda_vai_round(num_ele, input, output, method);
  }else if(device_id == 1){
    cpu_vai_round(num_ele, input, output, method);
  } else {
    LOG(FATAL) << "Unsupported tensor type: " << Tinput.toString();
  }
}

template
void _Round<float>(Tensor Tinput, Tensor Toutput, int64_t method, int64_t device_id);

template
void _Round<double>(Tensor Tinput, Tensor Toutput, int64_t method, int64_t device_id);

void Round(Tensor Tinput, Tensor Toutput, int64_t method, int64_t device_id){
  if (Tinput.dtype() == at::kFloat)
    _Round<float>(Tinput, Toutput, method, device_id);
  else if (Tinput.dtype() == at::kDouble)
    _Round<double>(Tinput, Toutput, method, device_id);
}

template <typename Dtype>
void _FixNeuronV2(Tensor Tinput,
                 Tensor Toutput,
                 int64_t valmin,
                 int64_t valmax,
                 Dtype valamp,
                 int64_t zero_point,
                 int64_t method,
                 int64_t device_id){

    auto input = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();
    if(device_id == 0){
      cuda_fix_neuron_v2(num_ele,
                         input,
                         output,
                         valmin,
                         valmax,
                         valamp,
                         zero_point,
                         1, //keep_scale
                         method);
    }
    else if(device_id == 1){
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
}

void FixNeuronV2(Tensor Tinput,
                 Tensor Toutput,
                 int valmin,
                 int valmax,
                 float valamp,
                 int zero_point,
                 int method,
                 int device_id){
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

at::Tensor fix_neuron_per_channel(at::Tensor Tinput, int64_t valmin, int64_t valmax, at::Tensor scale, at::Tensor zero_point, int64_t axis, int64_t method, int64_t device_id, int64_t inplace){
  if (!(Tinput.dtype() == at::kFloat) && !(Tinput.dtype() == at::kDouble))
    LOG(FATAL) << "Unsupported tensor type: " << Tinput.toString();

  std::vector<at::Tensor> Tinput_split = at::split(Tinput, 1, axis);
  if (inplace != 0) {
    for(int i=0; i<Tinput_split.size(); ++i) {
      float scale_i = *(scale[i].data<float>());
      double valamp_i = 1.0/double(scale_i);
      int64_t zero_point_i = int64_t(*(zero_point[i].data<int8_t>()));
      if (Tinput.dtype() == at::kFloat) {
        _FixNeuronV2<float>(Tinput_split[i],
                            Tinput_split[i],
                            valmin,
                            valmax,
                            valamp_i,
                            zero_point_i,
                            method,
                            device_id);
      } else if (Tinput.dtype() == at::kDouble) {
        _FixNeuronV2<double>(Tinput_split[i],
                             Tinput_split[i],
                             valmin,
                             valmax,
                             valamp_i,
                             zero_point_i,
                             method,
                             device_id);
      }
    }
    at::Tensor Toutput = at::cat(at::TensorList(Tinput_split), axis);
    return Toutput;
  } else {
    std::vector<at::Tensor> Toutput_vector;
    for(int i=0; i<Tinput_split.size(); ++i) {
      at::Tensor Toutput_i = at::native::empty_like(Tinput_split[i]);
      float scale_i = *(scale[i].data<float>());
      double valamp_i = 1.0/double(scale_i);
      int64_t zero_point_i = int64_t(*(zero_point[i].data<int8_t>()));
      if (Tinput.dtype() == at::kFloat) {
        _FixNeuronV2<float>(Tinput_split[i],
                            Toutput_i,
                            valmin,
                            valmax,
                            valamp_i,
                            zero_point_i,
                            method,
                            device_id);
      } else if (Tinput.dtype() == at::kDouble) {
        _FixNeuronV2<double>(Tinput_split[i],
                             Toutput_i,
                             valmin,
                             valmax,
                             valamp_i,
                             zero_point_i,
                             method,
                             device_id);
      }
      Toutput_vector.push_back(Toutput_i);
    }
    at::Tensor Toutput = at::cat(at::TensorList(Toutput_vector), axis);
    return Toutput;
  }
}
