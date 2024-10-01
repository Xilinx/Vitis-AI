

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
#include "../../include/nndct_diffs_op.h"
#include "c10/util/ArrayRef.h"
#include "../../../../../include/cpu/nndct_fix_kernels_cpu.h"
//#include <iostream>

template <typename Dtype>
void _DiffsFixPos(Tensor Tinput, 
                  Tensor Tbuffer,
                  Tensor Tfixpos,
                  int bit_width, 
                  int range, 
                  int method,
                  int device_id) {
    auto input = Tinput.data<Dtype>();
    auto buffer = Tbuffer.data<Dtype>();
    auto fixpos = Tfixpos.data<Dtype>();
    int64_t num_ele = Tinput.numel();
      
    cpu_diff_S( num_ele, 
                input, 
                buffer, 
                fixpos, 
                bit_width, 
                range, 
                method);
}

void DiffsFixPos(Tensor Tinput, 
                 Tensor Tbuffer,
                 Tensor Tfixpos,
                 int bit_width, 
                 int range, 
                 int method,
                 int device_id) {
  if (Tinput.dtype() == at::kFloat){
    _DiffsFixPos<float>(Tinput, 
                        Tbuffer,
                        Tfixpos,
                        bit_width, 
                        range, 
                        method,
                        device_id);
  }
  else if (Tinput.dtype() == at::kDouble){
    _DiffsFixPos<double>(Tinput, 
                         Tbuffer,
                         Tfixpos,
                         bit_width, 
                         range, 
                         method,
                         device_id);
  }
}


// c++ warapper for registration in torch script
void diffs_fix_pos(at::Tensor Tinput, at::Tensor Tbuffer, at::Tensor Tfixpos, int64_t bit_width, int64_t range, int64_t method, int64_t device_id){
  if (Tinput.dtype() == at::kFloat){
    _DiffsFixPos<float>(Tinput, 
                        Tbuffer,
                        Tfixpos,
                        bit_width, 
                        range, 
                        method,
                        device_id);
  }
  else if (Tinput.dtype() == at::kDouble){
    _DiffsFixPos<double>(Tinput, 
                         Tbuffer,
                         Tfixpos,
                         bit_width, 
                         range, 
                         method,
                         device_id);
  }
}

