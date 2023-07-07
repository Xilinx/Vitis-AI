

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


#ifndef NNDCT_FIXNEURON_OP_H
#define NNDCT_FIXNEURON_OP_H

using at::Tensor;
void FixNeuronV2 (Tensor Tinput, 
                  Tensor Toutput,
                  int valmin,
                  int valmax, 
                  float valamp, 
                  int zero_point,
                  int method,
                  int device_id);
at::Tensor fix_neuron(at::Tensor Tinput,
                      int64_t valmin,
                      int64_t valmax, 
                      double valamp, 
                      int64_t zero_point,
                      int64_t method, 
                      int64_t device_id, 
                      int64_t inplace);
#endif
