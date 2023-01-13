

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


#ifndef _NNDCT_CU_KERNELS_ANSI_H_
#define _NNDCT_CU_KERNELS_ANSI_H_

#ifdef __cplusplus
extern "C" {
#endif
//implemented in nndct_math_kernels.cu
void cudaF_partial_sort(float* host_in,int dim,float* host_out);

void cudaI_test_op(const int N,const int* input,int* out);
void cudaF_test_op(const int N,const float* input,float* out);

void cudaF_set(const int n,float* data,float val);
void cudaD_set(const int n,double* data,double val);
void cudaI_set(const int n,int* data,int val);

void cudaF_scale_inplace(const int n,float* data,float scale);
void cudaD_scale_inplace(const int n,double* data,double scale);
void cudaI_int_scale_inplace(const int n,int* data,int bitwidth,int fragpos);

void cudaF_scale(const int n,const float* src,float* dst,float scale);
void cudaD_scale(const int n,const double* src,double* dst,double scale);

void cudaF_mat_scale_inplace(int row,int col,int stride,float* data,float scale);
void cudaD_mat_scale_inplace(int row,int col,int stride,double* data,double scale);

void cudaF_pow(const int n,float* data,float pow);
void cudaD_pow(const int n,double* data,double pow);

void cudaF_max(const int n,const float* src,float* dst);
void cudaD_max(const int n,const double* src,double* dst);

void cudaF_min(const int n,const float* src,float* dst);
void cudaD_min(const int n,const double* src,double* dst);

void cudaF_sum(const int n,const float* src,float* dst);
void cudaD_sum(const int n,const double* src,double* dst);

void cudaF_max_inplace(const int n,float* data);
void cudaD_max_inplace(const int n,double* data);

void cudaF_min_inplace(const int n,float* data);
void cudaD_min_inplace(const int n,double* data);

void cudaF_sum_inplace(const int n,float* data);
void cudaD_sum_inplace(const int n,double* data);

void cudaF_sub(const int n,const float* src,float* dst);
void cudaD_sub(const int n,const double* src,double* dst);

void cudaF_flip_row_2d(int row,int col,float* out,const float* in,
  int gap,int src_stride);
void cudaD_flip_row_2d(int row,int col,double* out,const double* in,
  int gap,int src_stride);
void cudaI_flip_row_2d(int row,int col,int* out,const int* in,
  int gap,int src_stride);

void cudaF_vec_vec_element(int dim,const float* a,
  const float* b,float* dst,float alpha);
void cudaD_vec_vec_element(int dim,const double* a,
  const float* b,double* dst,double alpha);

void cudaI_addmm_dimi(int* c,const int c_dim,
  const int* a,const int a_dim,const int* b,const int b_dim,
  int trans_a,int trans_b,int m,int n,int k,int alpha,
  int beta,int bitwidth,int fragpos);

void cudaF_set_vec(float* dst,const float* src,int dim,float alpha);
void cudaD_set_vec(double* dst,const double* src,int dim,double alpha);
void cudaI_set_vec(int* dst,const int* src,int dim,int alpha);

void cudaF_add_vec(float* dst,const float* src,int dim,float alpha);
void cudaD_add_vec(double* dst,const double* src,int dim,double alpha);
void cudaI_add_vec(int* dst,const int* src,int dim,int alpha);

void cudaF_set_mat(int row,int col,float* dst,int dst_stride,
  const float* src,int src_stride,float alpha,int trans_int);
void cudaD_set_mat(int row,int col,double* dst,int dst_stride,
  const double* src,int src_stride,double alpha,int trans_int);
void cudaI_set_mat(int row,int col,int* dst,int dst_stride,
  const int* src,int src_stride,int alpha,int trans_int);

void cudaF_add_mat(int row,int col,float* dst,int dst_stride,
  const float* src,int src_stride,float alpha,int trans_int);
void cudaD_add_mat(int row,int col,double* dst,int dst_stride,
  const double* src,int src_stride,double alpha,int trans_int);
void cudaI_add_mat(int row,int col,int* dst,int dst_stride,
  const int* src,int src_stride,int alpha,int trans_int);

void cudaF_add_vec_rows(const float alpha, const float* vec, const float beta, 
  const float* src,float* dst, int row,int col,int stride);
void cudaD_add_vec_rows(const double alpha, const double* vec, const double beta, 
  const double* src,double* dst, int row,int col,int stride);

void cudaF_add_vec_rows_inplace(const float alpha, const float* vec, const float beta, 
  float* mat, int row,int col,int stride);
void cudaD_add_vec_rows_inplace(const double alpha, const double* vec, const double beta, 
  double* mat, int row,int col,int stride);
void cudaI_add_vec_rows_inplace(const int alpha, const int* vec, const int beta, 
  int* mat, int row,int col,int stride);

void cudaF_add_vec_cols(const float alpha, const float* vec, const float beta, 
  const float* src,float* dst, int row,int col,int stride);
void cudaD_add_vec_cols(const double alpha, const double* vec, const double beta, 
  const double* src,double* dst, int row,int col,int stride);

void cudaF_add_vec_cols_inplace(const float alpha, const float* vec, const float beta, 
  float* mat, int row,int col,int stride);
void cudaD_add_vec_cols_inplace(const double alpha, const double* vec, const double beta, 
  double* mat, int row,int col,int stride);
void cudaI_add_vec_cols_inplace(const int alpha, const int* vec, const int beta, 
  int* mat, int row,int col,int stride);

void cudaI_add_vec_rows_inplace_dimi(const int alpha, const int* vec, const int beta, 
  int* mat, int row,int col,int stride,float multiplier);

//implemented in nndct_fixpoint_kernels.h
void cudaF_scale_T2int_mat(int row,int col,int* out_data,
  const float* in_data,int in_stride,int out_stride,float m);
void cudaD_scale_T2int_mat(int row,int col,int* out_data,
  const double* in_data,int in_stride,int out_stride,double m);

void cudaF_scale_int2T(int n,float* out_data,const int* in_data,float m);
void cudaD_scale_int2T(int n,double* out_data,const int* in_data,double m);

void cudaF_scale_int2T_mat(int row,int col,float* out_data,
  const int* in_data,int in_stride,int out_stride,float m);
void cudaD_scale_int2T_mat(int row,int col,double* out_data,
  const int* in_data,int in_stride,int out_stride,double m);

void cudaF_scale_T2int_vec(int dim,int* out_data,const float* in_data,float m);
void cudaD_scale_T2int_vec(int dim,int* out_data,const double* in_data,double m);

void cudaF_scan_maxmin(int row,int col,int src_stride, int scan_stride,
  const float* src,float* max,float* min);
void cudaD_scan_maxmin(int row,int col,int src_stride, int scan_stride,
  const double* src,double* max,double* min);

//void cudaF_fix_neuron_v2(const int N,const float* src,
//  float* dst,int val_max,float val_amp,int keep_scale,int method);
void cudaD_fix_neuron_v2(const int N,const double* src,
  double* dst,int val_max,double val_amp,int keep_scale,int method);

void cudaF_fix_neuron_v2_inplace(const int N,float* data,
  int val_max,float val_amp,int keep_scale,int method);
void cudaD_fix_neuron_v2_inplace(const int N,double* data,
  int val_max,double val_amp,int keep_scale,int method);

void cudaF_fix_neuron_v2_int(const int N,const float* src,
  int* dst,int bitwidth,int fragpos,int method);
void cudaD_fix_neuron_v2_int(const int N,const double* src,
  int* dst,int bitwidth,int fragpos,int method);

//implemented in nndct_prune_kernels.h
void cudaF_transpose_by_pe(int row,int col,int dst_col,
  const float* src,float* dst,int pe_num);
void cudaD_transpose_by_pe(int row,int col,int dst_col,
  const double* src,double* dst,int pe_num);

void cudaF_fill_by_sorted_pe(int row,int col,int sort_col,
  float* src,const float* sort,int pe_num,int key_idx);
void cudaD_fill_by_sorted_pe(int row,int col,int sort_col,
  double* src,const double* sort,int pe_num,int key_idx);

//implemented in nndct_conv_kernels.cu
void cudaF_im2col(const int row,const int col,const float* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  float* data_buf,const int g_start,const int batch_first);
void cudaD_im2col(const int row,const int col,const double* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  double* data_buf,const int g_start,const int batch_first);
void cudaI_im2col(const int row,const int col,const int* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  int* data_buf,const int g_start,const int batch_first);

void cudaF_col2im(const int row,const int col,float* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  const float* data_buf,
  const int bz_start,const int bz_end,
  const int k_idx_start,const int k_idx_end,
  const int g_stride,const int batch_first);
void cudaD_col2im(const int row,const int col,double* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  const double* data_buf,
  const int bz_start,const int bz_end,
  const int k_idx_start,const int k_idx_end,
  const int g_stride,const int batch_first);

void cudaF_col2im_fast(const int row,const int col,float* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  const float* data_buf,const int g_start,const int batch_first);
/*void cudaD_col2im_fast(const int row,const int col,double* data_im, 
  const int batch_size,const int channels,const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int stride_h, const int stride_w,
  const int height_col, const int width_col,
  const double* data_buf,const int g_start,const int batch_first);*/

//implemented in nndct_batchnorm_kernels.cu
void cudaF_batch_norm_inference(int row,int col,float eps,
  const float* input,const float* gamma,const float* beta,
  const float* mean,const float* var,float* out);
void cudaD_batch_norm_inference(int row,int col,float eps,
  const double* input,const double* gamma,const double* beta,
  const double* mean,const double* var,double* out);

void cudaF_batch_norm_training(int row,int col,float eps,
  int renorm,const float* input,const float* gamma,
  const float* beta,const float* mean,const float* var,float* out);
void cudaD_batch_norm_training(int row,int col,float eps,
  int renorm,const double* input,const double* gamma,
  const double* beta,const double* mean,const double* var,double* out);

void cudaF_batch_norm_scan(int row,int col,float eps,
  const float* input,const float* gamma,const float* beta,
  const float* mean,const float* var,float* out,float* max,float* min);
void cudaD_batch_norm_scan(int row,int col,float eps,
  const double* input,const double* gamma,const double* beta,
  const double* mean,const double* var,double* out,double* max,double* min);

void cudaF_batch_norm_quant(int row,int col,float eps,
  const float* input,const float* gamma,const float* beta,
  const float* mean,const float* var,float* out,
  const int parammaxA,const int parammaxB,
  const float paramampA,const float paramampB,
  const int in_max,const float in_amp,int keep_scale);
void cudaD_batch_norm_quant(int row,int col,double eps,
  const double* input,const double* gamma,const double* beta,
  const double* mean,const double* var,double* out,
  const int parammaxA,const int parammaxB,
  const double paramampA,const double paramampB,
  const int in_max,const double in_amp,int keep_scale);

void cudaF_batch_norm_back_ivar(int row,int col,
  const float* out_grad,const float* input,const float* mean,
  const float* gamma,float* grad_buff);
void cudaD_batch_norm_back_ivar(int row,int col,
  const double* out_grad,const double* input,const double* mean,
  const double* gamma,double* grad_buff);

void cudaF_batch_norm_back(int row,int col,float eps,int renorm,
  const float* out_grad,const float* input,const float* gamma,
  const float* mean,const float* var,const float* divar,
  float* in_grad,float* grad_buff);
void cudaD_batch_norm_back(int row,int col,float eps,int renorm,
  const double* out_grad,const double* input,const double* gamma,
  const double* mean,const double* var,const double* divar,
  double* in_grad,double* grad_buff);

//implemented in nndct_lstm_kernels.cu
void cudaF_lstm_cell(const int N,const int num_cell,const int stride,
  const float* y_c_prev,float* y_g,float* y_i,float* y_f,float* y_o,
  float* y_c,float* y_h,float cell_clip);
void cudaD_lstm_cell(const int N,const int num_cell,const int stride,
  const double* y_c_prev,double* y_g,double* y_i,double* y_f,double* y_o,
  double* y_c,double* y_h,double cell_clip);

void cudaF_lstm_cell_scan(int row,int col,const int gifo_stride,
  const float* y_c_prev,float* y_g,float* y_i,float* y_f,float* y_o,
  float* y_c,float* y_h,float* bmax,float* bmin,float cell_clip);
void cudaD_lstm_cell_scan(int row,int col,const int gifo_stride,
  const double* y_c_prev,double* y_g,double* y_i,double* y_f,double* y_o,
  double* y_c,double* y_h,double* bmax,double* bmin,double cell_clip);

void cudaF_lstm_cell_quant(int row,int col,const int gifo_stride,
  const float* y_c_prev,float* y_g,float* y_i,float* y_f,float* y_o,
  float* y_c,float* y_h,float cell_clip,int valmax_out,float valamp_out,
  int sigmoid_max,int tanh_max,const int* sigmoid_data,const int* tanh_data,
  int keep_scale,int debug);
void cudaD_lstm_cell_quant(int row,int col,const int gifo_stride,
  const double* y_c_prev,double* y_g,double* y_i,double* y_f,double* y_o,
  double* y_c,double* y_h,double cell_clip,int valmax_out,double valamp_out,
  int sigmoid_max,int tanh_max,const int* sigmoid_data,const int* tanh_data,
  int keep_scale,int debug);

void cudaI_lstm_cell_int(int row,int col,const int gifo_stride,
  const int* y_c_prev,int* y_g,int* y_i,int* y_f,int* y_o,
  int* y_c,int* y_h,int cell_clip,int out_bn,int out_fp,
  int sigmoid_expo,int sigmoid_fp,int tanh_expo,int tanh_fp,
  const int* sigmoid_data,const int* tanh_data,int debug);

void cudaF_lstm_back_cell(int row,int col,const int gifo_stride, 
  const float* y_g,const float* y_i,const float* y_f,const float* y_o,
  const float* y_h,const float* prev_y_c,float* d_g,float* d_i,
  float* d_f,float* d_o,float* d_h,float* d_c,float* d_c_next,
  float diff_clip,float cell_diff_clip);
void cudaD_lstm_back_cell(int row,int col,const int gifo_stride, 
  const double* y_g,const double* y_i,const double* y_f,const double* y_o,
  const double* y_h,const double* prev_y_c,double* d_g,double* d_i,
  double* d_f,double* d_o,double* d_h,double* d_c,double* d_c_next,
  double diff_clip,double cell_diff_clip);

void cudaF_diff_S(const int N, const float* src, float* buffer, float* output, int bitwidth, int range, int method);

void cudaD_diff_S(const int N, const double* src, double* buffer, double* output, int bitwidth, int range, int method);

#ifdef __cplusplus
} //extern "C"
#endif
#endif //_NNDCT_CU_KERNELS_ANSI_H_
