

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


#ifndef NNDCT_MATH_GPU_H
#define NNDCT_MATH_GPU_H

using at::Tensor;

void Scale(Tensor Tinput, float scale, int device_id);

void SigmoidTableLookup(Tensor Tinput, 
                        Tensor Ttable, 
                        Tensor Toutput, 
                        int fragpos,
                        int device_id);

void SigmoidSimulation(Tensor Tinput, 
                        Tensor Toutput, 
                        int device_id);

void TanhTableLookup(Tensor Tinput, 
                     Tensor Ttable, 
                     Tensor Toutput, 
                     int fragpos,
                     int device_id);

void TanhSimulation(Tensor Tinput, 
                     Tensor Toutput, 
                     int device_id);

#endif
