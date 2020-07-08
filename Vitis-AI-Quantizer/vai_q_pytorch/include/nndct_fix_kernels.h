#ifndef _NNDCT_FIX_KERELS_H_
#define  NNDCT_FIX_KERELS_H_

template<typename Dtype>
void cuda_sigmoid_table_lookup(const int N, 
                               const Dtype* input, 
                               const Dtype* table,
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
                        int val_max, 
                        int keep_scale, 
                        int method);

template<typename Dtype>
void cuda_fix_neuron_v2(const int N, 
                        const Dtype* src,
                        Dtype* dst, 
                        int val_max, 
                        Dtype val_amp, 
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

#endif //_NNDCT_CU_KERNELS_ANSI_H_
