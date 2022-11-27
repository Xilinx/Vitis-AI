

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



#ifndef NNDCT_TORCH_DIFFS_OP_H
#define NNDCT_TORCH_DIFFS_OP_H

using at::Tensor;

void DiffsFixPos(Tensor Tinput, 
                 Tensor Tbuffer,
                 Tensor Tfixpos, 
                 int bit_width, 
                 int range, 
                 int method,
                 int device_id); 

void diffs_fix_pos(at::Tensor Tinput, at::Tensor Tbuffer, at::Tensor Tfixpos, int64_t bit_width, int64_t range, int64_t method, int64_t device_id);

#endif // NNDCT_TORCH_DIFFS_OP_H
