

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

#ifndef _NNDCT_FIX_KERELS_H_
#define _NNDCT_FIX_KERELS_H_

template<typename Dtype>
void cuda_sigmoid_table_lookup(const int N, 
                               const Dtype* input, 
                               const Dtype* table,
                               Dtype* output,
                               int fragpos);  

template<typename Dtype>
void cuda_sigmoid_simulation(const int N, 
                               const Dtype* input, 
                               Dtype* output,
                               int fragpos);  

template<typename Dtype>
void cuda_tanh_simulation(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            int fragpos);  

template<typename Dtype>
void cuda_tanh_table_lookup(const int N, 
                            const Dtype* input, 
                            const Dtype* table,
                            Dtype* output,
                            int fragpos);  

template<typename Dtype>
void cuda_fix_neuron_v1(const int N, 
                        const Dtype* src,
                        const Dtype* fragpos, 
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        int keep_scale, 
                        int method);

template<typename Dtype>
void cuda_fix_neuron_v2(const int N, 
                        const Dtype* src,
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        Dtype val_amp, 
                        int zero_point,
                        int keep_scale, 
                        int method);

template<typename Dtype>
void cuda_diff_S(const int N, 
                 const Dtype* src, 
                 Dtype* buffer, 
                 Dtype* output, 
                 int bitwidth, 
                 int range, 
                 int method);

template<typename Dtype>
void cuda_softmax_exp_approximate(const int N,
                            const Dtype* input,
                            Dtype* output);  

template<typename Dtype>
void cuda_softmax_lod(const int N,
                            const Dtype* input,
                            Dtype* output);  

template<typename Dtype>
void cuda_softmax_simulation_part_1(const int N,
                            const Dtype* input,
                            Dtype* output); 

template<typename Dtype>
void cuda_softmax_simulation_part_2(const int N,
                            const Dtype* sum,
                            Dtype* output);  

template<typename Dtype>
void cuda_sigmoid_table_lookup_aie2(const int N, 
                               const Dtype* input, 
                               Dtype* output,
                               int fragpos);

template<typename Dtype>
void cuda_tanh_table_lookup_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            int fragpos); 

template<typename Dtype>
void cuda_exp_appr_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            const int bit_width);  
template<typename Dtype>
void cuda_log_softmax_fast_ln(const int N,
                            const Dtype* input,
                            Dtype* output);  

template<typename Dtype>
void cuda_log_softmax_sub(const int N,
                            const Dtype* input,
                            Dtype* output,
                            const Dtype* sub);  

template<typename Dtype>
void cuda_layernorm_isqrt(const int N,
                            const Dtype* input,
                            Dtype* output); 

template<typename Dtype>
void cuda_layernorm_invsqrt(const int N,
                            const Dtype* input,
                            Dtype* output); 
template<typename Dtype>
void cuda_inverse_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output); 

#endif //_NNDCT_FIX_KERELS_H_
