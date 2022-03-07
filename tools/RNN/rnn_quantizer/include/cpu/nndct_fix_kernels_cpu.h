

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

#include <stdio.h>
#include <stdlib.h>

#ifndef _NNDCT_FIX_KERELS_CPU_H_
#define  NNDCT_FIX_KERELS_CPU_H_

template<typename Real>
void _fix_neuron_v2_cpu(const Real& src,int& res,
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
void _mapping_sigm_cpu(const Real output_amp,
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
void _mapping_tanh_cpu(const Real output_amp,
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
void _mappingI_sigm_cpu(const Real output_fp,
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
void _mappingI_tanh_cpu(const Real output_fp,
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
void _scaleI_cpu(Real &result,int bitwidth,int shift){
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

//check if this cpu kenrnels are needed
template<typename Real>
void _dimi_floor_cpu(Real &result,
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
void _amp_floor_cpu(Real &result,const Real val_amp,const int val_max){
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
void _dimi_cpu(Real &result,const Real val_amp){
  result /= val_amp;
}

template<typename Real>
void _amp_cpu(Real &result,const Real val_amp){
  result *= val_amp;
}

template<typename Real>
void _floor_cpu(Real &result,const int val_max){
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
void _dimiI_cpu(int &result,Real diff_amp){
  int tmp_result = diff_amp>=1? int(result/diff_amp): int(result*diff_amp);
  if(diff_amp>1 && result%(int)diff_amp!=0 &&result<0){
    tmp_result-=1;
  }
  result=tmp_result;
}

template<typename Real>
void _dimiI_floor_cpu(Real &result,
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
void _fix_neuron_v2_cpu_tmp(Real& result,Real val_amp,
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


template<typename Dtype>
void cpu_sigmoid_table_lookup(const int N, 
                              const Dtype* input, 
                              const Dtype* table,
                              Dtype* output,
                              int fragpos);  

template<typename Dtype>
void cpu_tanh_table_lookup(const int N, 
                           const Dtype* input, 
                           const Dtype* table,
                           Dtype* output,
                           int fragpos);  

template<typename Dtype>
void cpu_fix_neuron_v1(const int N, 
                       const Dtype* src,
                       const Dtype* fragpos, 
                       Dtype* dst, 
                       int val_max, 
                       int keep_scale, 
                       int method);

template<typename Dtype>
void cpu_fix_neuron_v2(const int N, 
                       const Dtype* src,
                       Dtype* dst, 
                       int val_max, 
                       Dtype val_amp, 
                       int keep_scale, 
                       int method);

template<typename Dtype>
void cpu_diff_S(const int N, 
                const Dtype* src, 
                Dtype* buffer, 
                Dtype* output, 
                int bitwidth, 
                int range, 
                int method);

#endif //_NNDCT_FIX_KERNELS_CPU_H_
