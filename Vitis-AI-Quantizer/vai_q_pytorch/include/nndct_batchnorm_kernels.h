

/* 
# (c) Copyright 2016 – 2019 Xilinx, Inc. All rights reserved. 
# 
# This file contains confidential and proprietary information 
# of Xilinx, Inc. and is protected under U.S. and 
# international copyright and other intellectual property
# laws.
# 
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE;
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
# 
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
# 
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES
*/


#ifndef _NNDCT_BATCHNORM_KERNELS_H_
#define _NNDCT_BATCHNORM_KERNELS_H_


#ifdef __cplusplus
extern "C"{
#endif 
void cudaF_batch_norm_inference(int row,int col,float eps,
  const float* input,const float* gamma,const float* beta,
  const float* mean,const float* var,float* out);

void cuda_batch_norm_training(int row,int col,float eps,
  int renorm,const float* input,const float* gamma,
  const float* beta,const float* mean,const float* var,float* out);

void cudaF_batch_norm_scan(int row,int col,float eps,
  const float* input,const float* gamma,const float* beta,
  const float* mean,const float* var,float* out,float* max,float* min);

void cudaF_batch_norm_quant(int row,int col,float eps,
  const float* input,const float* gamma,const float* beta,
  const float* mean,const float* var,float* out,
  const int parammaxA,const int parammaxB,
  const float paramampA,const float paramampB,
  const int in_max,const float in_amp, int keep_scale);

void cudaF_batch_norm_back_ivar(int row,int col,
  const float* out_grad,const float* input,const float* mean,
  const float* gamma,float* grad_buff);

void cudaF_batch_norm_back(int row,int col,float eps,int renorm,
  const float* out_grad,const float* input,const float* gamma,
  const float* mean,const float* var,const float* divar,
  float* in_grad,float* grad_buff);
#ifdef __cplusplus
} //extern "C"
#endif

#endif //_NNDCT_BATCHNORM_KERNELS_H_
