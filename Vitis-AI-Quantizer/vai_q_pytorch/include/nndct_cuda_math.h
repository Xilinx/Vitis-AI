
#ifndef _NNDCT_CUDA_MATH_H_
#define _NNDCT_CUDA_MATH_H_

template<typename Dtype>
void cuda_set(const int n, Dtype* data, Dtype val);

template<typename Dtype>
void cuda_max(const int n, const Dtype* src, Dtype* dst);

template<typename Dtype>
void cuda_pow(const int n, Dtype* data, Dtype pow);

template<typename Dtype>
void cuda_min(const int n, const Dtype* src, Dtype* dst);

template<typename Dtype>
void cuda_sub(const int n, const Dtype* src, Dtype* dst);

template<typename Dtype>
void cuda_sum(const int n, Dtype* src, Dtype* dst);

template<typename Dtype>
void cuda_sum_inplace(const int n, Dtype* data);

template<typename Dtype>
void cuda_scale_inplace(const int n, 
                        Dtype* data, 
                        Dtype scale);

template<typename Dtype>
void cuda_scale(const int n, 
                const Dtype* src, 
                Dtype* dst, 
                Dtype scale);
#endif //_NNDCT_CUDA_MATH_H_

