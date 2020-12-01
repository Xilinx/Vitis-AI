

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

#ifndef _NNDCT_CPU_MATH_H_
#define _NNDCT_CPU_MATH_H_

template<typename Dtype>
void cpu_set(const int n, Dtype* data, Dtype val);

template<typename Dtype>
void cpu_max(const int n, const Dtype* src, Dtype& dst);

template<typename Dtype>
void cpu_pow(const int n, Dtype* data, Dtype pow);

template<typename Dtype>
void cpu_min(const int n, const Dtype* src, Dtype& dst);

template<typename Dtype>
void cpu_sub(const int n, const Dtype* src, Dtype* dst);

template<typename Dtype>
void cpu_sum(const int n, const Dtype* src, Dtype& dst);

template<typename Dtype>
void cpu_scale_inplace(const int n, 
                       Dtype* data, 
                       Dtype scale);

template<typename Dtype>
void cpu_scale(const int n, 
               const Dtype* src, 
               Dtype* dst, 
               Dtype scale);

#endif //_NNDCT_CPU_MATH_H_

