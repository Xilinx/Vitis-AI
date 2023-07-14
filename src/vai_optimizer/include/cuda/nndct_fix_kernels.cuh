

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


#ifndef _NNDCT_FIX_KERNELS_CUH_
#define _NNDCT_FIX_KERNELS_CUH_

template<typename Real>
__device__ void _vai_round_device(const Real& x, int& y, int method){
  if(2==method){  //half_up
    if(x<0 && (x-floor(x))==0.5) {
      y = ceil(x);
    }else{
      y = round(x);
    }
  }else if(3==method){ //c++ std::round: negative half_down, positive half_up
    y = round(x);
  }else if(4==method){ // floor
    y = floor(x);
  }else if(5==method){ // negative half_up, positive half_even 
    if(x<0 && (x-floor(x))==0.5) {
      y = ceil(x);
    }else if(x - floor(x) == 0.5){
      if( int(floor(x)) % 2 == 0){
        y = floor(x);
      }else{
        y = ceil(x);
      }
    }else{
      y = round(x);
    }
  }else if(6==method){ // towards zero: negative half_up, positive half_down (vs method 3)
    if(x<0 && (x-floor(x))==0.5) {
      y = ceil(x);
    }else if(x>0 && (x-floor(x))==0.5){
      y = floor(x);
    }else{
      y = round(x);
    }
  }else if(7==method){ // up
    y = ceil(x);
  }else if(8==method){ // half_even
    if(x<0 && (x-floor(x))==0.5) {
      if(int(ceil(x)) % 2 == 0){
        y = ceil(x);
      }else{
        y = floor(x);
      }
    }else if(x - floor(x) == 0.5) {
      if( int(floor(x)) % 2 == 0){
        y = floor(x);
      }else{
        y = ceil(x);
      }
    }else{
      y = round(x);
    }
  }
}

template<typename Real>
__device__ void _fix_neuron_v2_device(const Real& src,int& res,
  int val_min, int val_max,Real val_amp,int zero_point,int method){
  Real res_real_= src*val_amp;
  //method: 
  // 2: half_up 
  // 3: c++ std::round: negative half_down, positive half_up
  // 4: floor
  // 5: negative half_up, positive half_even
  // 6: towards zero: negative half_up, positive half_down (vs method 3)
  // 7: up
  // 8: half_even 
  _vai_round_device(res_real_, res, method);

  res = res + zero_point;
  if (res > val_max) {
      res = val_max;
  } else if (res < val_min) {
      res=val_min;
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
  const Real val_amp,const int val_min,const int val_max){

  int result_= floor(result/val_amp);
  if (result_ > val_max) {
    //result_ %=val_max;
    result_ = val_max;
  } else if (result_ < val_min) {
    //result_ %= -val_max;
    result_ = val_min;
  }
  result=(Real)(result_);
}

template<typename Real>
__device__ void _amp_floor_device(Real &result,const Real val_amp,const int val_min,const int val_max){
  int result_= floor(result*val_amp);
  if (result_ > val_max) {
    result_ = val_max;
  } else if (result_ < val_min) {
    result_ = val_min;
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
__device__ void _floor_device(Real &result,const int val_min,const int val_max){
  int result_ = floor(result);
  if (result_ > val_max) {
    result_ = val_max;
  } else if (result_ < val_min) {
    result_ = val_min;
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
  const Real val_amp,const Real val_min,const Real val_max){
  result/=val_amp;
  if (result > val_max) {
    result = val_max;
  } else if (result < val_min) {
    result = val_min;
  }
}

template<typename Real>
__device__ void _fix_neuron_v2_device_tmp(Real& result,Real val_amp,
  int val_min,int val_max,bool dimi,bool keep_scale,int method){
  if(0==method){
    result=(!dimi)? result*val_amp:result*(1/val_amp);
  }else if(1==method || 3==method){
    int result_;
    if(1==method){
      result_= (!dimi)? floor(result*val_amp):floor(result*(1/val_amp));
    }else{
      result_= (!dimi)? ceil(result*val_amp):ceil(result*(1/val_amp));
    }
    if (result_ > val_max) {
      result_ = val_max;
    } else if (result_ < val_min) {
      result_ = val_min;
    }
    if(keep_scale){
      result= (!dimi)? Real(result_)*(1/val_amp):Real(result_)*val_amp;
    }else{
      result=result_;
    }
  }else if(2==method){
    Real result_= (0==dimi)? result*val_amp:result*(1/val_amp);
    int fixed_result_;
    if(result_ > val_max) {
      result_ = val_max;
    }else if(result_ < val_min) {
      result_= val_min;
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
