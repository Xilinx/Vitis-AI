

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
#include "../include/nndct_diffs_op.h"
#include "c10/util/ArrayRef.h"
#include "../../../../include/nndct_fix_kernels.h"

template <typename Dtype>
void _DiffsFixPos(Tensor Tinput, 
                 Tensor Tbuffer,
                 Tensor Tfixpos,
                 int bit_width, 
                 int range, 
                 int method) {

    auto input = Tinput.data<Dtype>();
    auto buffer = Tbuffer.data<Dtype>();
    auto fixpos = Tfixpos.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    cuda_diff_S(num_ele, 
                input, 
                buffer, 
                fixpos, 
                bit_width, 
                range, 
                method ); 
}

void DiffsFixPos(Tensor Tinput, 
                 Tensor Tbuffer,
                 Tensor Tfixpos,
                 int bit_width, 
                 int range, 
                 int method) {
  if (Tinput.dtype() == at::kFloat)
    _DiffsFixPos<float>(Tinput, 
                        Tbuffer,
                        Tfixpos,
                        bit_width, 
                        range, 
                        method);
  else if (Tinput.dtype() == at::kDouble)
    _DiffsFixPos<double>(Tinput, 
                         Tbuffer,
                         Tfixpos,
                         bit_width, 
                         range, 
                         method);
}

