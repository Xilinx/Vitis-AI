

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


#ifndef _NNDCT_FIX_KERNELS_CUH_
#define _NNDCT_FIX_KERNELS_CUH_
template<typename Real>
__device__ void _fix_neuron_v2_device(const Real& src,int& res,
  int val_max,Real val_amp,int method){
  Real res_real_= src*val_amp;
  //method: 
  //5: old RNN
  //2: special round for -x.5, 
  //3: standard round
  //4: new RNN
  if(4==method){
    res = floor(res_real_);
    if (res > val_max-1) {
      res = val_max-1;
    } else if (res < -val_max) {
      res=-val_max;
    }
  }else if(2==method){  
    if(res_real_ > val_max-1) {
      res = val_max-1;
    }else if(res_real_ < -val_max) {
      res=-val_max;
    }else if(res_real_<0 && (res_real_-floor(res_real_))==0.5) {
      res = ceil(res_real_);
    }
    else{
      res = round(res_real_);
    }
  }else if(3==method){ 
    if(res_real_ > val_max-1) 
      res = val_max-1;
    else if(res_real_ < -val_max)
      res=-val_max;
    else
      res = round(res_real_);
  }else if(5==method){
    if(res_real_ > val_max-1) {
      res = val_max-1;
    }else if(res_real_ < -val_max) {
      res=-val_max;
    }else if(res_real_<0 && (res_real_-floor(res_real_))==0.5) {
      res = ceil(res_real_);
    }else if(res_real_ - floor(res_real_) == 0.5){
      if( int(floor(res_real_)) % 2 == 0){
        res = floor(res_real_);
      }else{
        res = ceil(res_real_);
      }
    }else{
      res = round(res_real_);
    }
  }
}

template<typename Real>
__device__ void _mapping_sigm_device(const Real output_amp,
  const int* map_data,Real &src,Real &dst){
  if (src>=8*output_amp){
    dst=32767;
  }else if (src<-8*output_amp){
    dst=0;
  }else {
    if(src >=0){
      int pos=0;
      if (output_amp>=128){
        pos = floor(src/(output_amp/128));
      }else {
        pos = floor(src*(128/output_amp));
      }
      pos%=1024;
      dst = map_data[1024+pos];
    }else{
      int pos=0;
      if (output_amp >= 128){
        pos = floor(fabs(src)/(output_amp/128));
      }else {
        pos = floor(fabs(src)*(128/output_amp));
      }
      pos%=1024;
      if((src == -8*output_amp) && pos == 0){
        dst = 0;
      } else {
        dst = map_data[1024-pos];
      }
    }
  }
}

template<typename Real>
__device__ void _mapping_tanh_device(const Real output_amp,
  const int* map_data,Real &src,Real &dst){
  if (src>=4*output_amp){
    dst=32767;
  }else if (src<-4*output_amp){
    dst=-32768;
  }else {
    if (src >= 0){
      int pos=0;
      if (output_amp>=256){
        pos = floor(src/(output_amp/256));
      }else {
        pos = floor(src*(256/output_amp));
      }
      pos%=1024;
      dst = map_data[1024 + pos];    
    } else {
      int pos = 0;
      if (output_amp>=256){
        pos = floor(fabs(src)/(output_amp/256));
      }else {
        pos = floor(fabs(src)*(256/output_amp));
      }
      pos%=1024;
      if((src == -4*output_amp) && (pos == 0)){
        dst = map_data[pos];
      } else {
        dst = map_data[1024 - pos];
      }
    }
  }
}

template<typename Real>
__device__ void _mappingI_sigm_device(const Real output_fp,
  const int* map_data,Real &src,Real &dst){
  if ((src>>output_fp)>=8){
    dst=32767;
  }else if ((src>>output_fp)<-8){
    dst=0;
  }else {
    int pos=0;
    if (output_fp>=7){
      pos = src >> (output_fp - 7);
    }else {
      pos = src << (7 - output_fp);
    }
    pos%=2048;
    if(pos<0){
      dst=map_data[2048+pos];
    }else{
      dst=map_data[pos];
    }
  }
}

template<typename Real>
__device__ void _mappingI_tanh_device(const Real output_fp,
  const int* map_data,Real &src,Real &dst){
  if ((src>>output_fp)>=4){
    dst=32767;
  }else if ((src>>output_fp)<-4){
    dst=-32768;
  }else {
    int pos=0;
    if (output_fp>=8){
      pos = src>>(output_fp-8);
    }else {
      pos = src<<(8-output_fp);
    }
    pos%=2048;
    if(pos<0){
      dst=map_data[2048+pos];
    }else{
      dst=map_data[pos];
    }
  }
}

template<typename Real>
__device__ void _scaleI_device(Real &result,int bitwidth,int shift){
  if(shift>0){
    result<<=shift;
  }else{
    result>>=(-shift);
  }
  int max_val=1<<bitwidth;
  if (result > max_val-1) {
    result=result%max_val-max_val;
  } else if (result < -max_val) {
    result=max_val+result%(-max_val);
  }
}

//check if this device kenrnels are needed
template<typename Real>
__device__ void _dimi_floor_device(Real &result,
  const Real val_amp,const int val_max){

  int result_= floor(result/val_amp);
  if (result_ > val_max-1) {
    //result_ %=val_max;
    result_ =val_max;
  } else if (result_ < -val_max) {
    //result_ %= -val_max;
    result_ = -val_max;
  }
  result=(Real)(result_);
}

template<typename Real>
__device__ void _amp_floor_device(Real &result,const Real val_amp,const int val_max){
  int result_= floor(result*val_amp);
  if (result_ > val_max-1) {
    //result_ %=val_max;
    result_ =val_max;
  } else if (result_ < -val_max) {
    //result_ %= -val_max;
    result_ = -val_max;
  }
  result=(Real)(result_);
}

template<typename Real>
__device__ void _dimi_device(Real &result,const Real val_amp){
  result /= val_amp;
}

template<typename Real>
__device__ void _amp_device(Real &result,const Real val_amp){
  result *= val_amp;
}

template<typename Real>
__device__ void _floor_device(Real &result,const int val_max){
  int result_ = floor(result);
  if (result_ > val_max-1) {
    //result_ %=val_max;
    result_ =val_max;
  } else if (result_ < -val_max) {
    //result_ %= -val_max;
    result_ = -val_max;
  }
  result=(Real)(result_);
}

template<typename Real>
__device__ void _dimiI_device(int &result,Real diff_amp){
  int tmp_result = diff_amp>=1? int(result/diff_amp): int(result*diff_amp);
  if(diff_amp>1 && result%(int)diff_amp!=0 &&result<0){
    tmp_result-=1;
  }
  result=tmp_result;
}

template<typename Real>
__device__ void _dimiI_floor_device(Real &result,
  const Real val_amp,const Real val_max){
  result/=val_amp;
  if (result > val_max-1) {
    //result %=val_max;
    result =val_max;
  } else if (result < -val_max) {
    //result %= -val_max;
    result = -val_max;
  }
}

template<typename Real>
__device__ void _fix_neuron_v2_device_tmp(Real& result,Real val_amp,
  int val_max,bool dimi,bool keep_scale,int method){
  if(0==method){
    result=(!dimi)? result*val_amp:result*(1/val_amp);
  }else if(1==method || 3==method){
    int result_;
    if(1==method){
      result_= (!dimi)? floor(result*val_amp):floor(result*(1/val_amp));
    }else{
      result_= (!dimi)? ceil(result*val_amp):ceil(result*(1/val_amp));
    }
    if (result_ > val_max-1) {
      //result_ %=val_max;
      result_ =val_max-1;
    } else if (result_ < -val_max) {
      //result_ %= -val_max;
      result_ = -val_max;
    }
    if(keep_scale){
      result= (!dimi)? Real(result_)*(1/val_amp):Real(result_)*val_amp;
    }else{
      result=result_;
    }
  }else if(2==method){
    Real result_= (0==dimi)? result*val_amp:result*(1/val_amp);
    int fixed_result_;
    if(result_ > val_max-1) {
      result_ = val_max-1;
    }else if(result_ < -val_max) {
      result_=-val_max;
    }else if(result_<0 && (result_-floor(result_))==0.5) {
      fixed_result_ = ceil(result_);
    } else {
      fixed_result_ = round(result_);
    }
    if(keep_scale){
      result= (!dimi)? Real(fixed_result_)*(1/val_amp):Real(fixed_result_)*val_amp;
    }else{
      result=fixed_result_;
    }
  }
}

#endif //_NNDCT_QUANTIZATION_KERNELS_CUH_
