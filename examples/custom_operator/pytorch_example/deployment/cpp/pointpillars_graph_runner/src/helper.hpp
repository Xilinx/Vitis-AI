/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <memory>
#include <queue>
#include <glog/logging.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "./pointpillars.hpp"

using namespace std;

namespace vitis { namespace ai { namespace pp{

using V3F = std::vector<V2F>;
using V4F = std::vector<V3F>;
using V5F = std::vector<V4F>;
using V6F = std::vector<V5F>;
using V7F = std::vector<V6F>;

using V3I=std::vector<V2I>;

// begin config file item 
extern V1F cfg_voxel_size;
extern V1F cfg_point_cloud_range;
extern V1I cfg_layer_strides;
extern V2F cfg_anchor_generator_stride_sizes;
extern V2F cfg_anchor_generator_stride_strides;
extern V2F cfg_anchor_generator_stride_offsets;
extern V2F cfg_anchor_generator_stride_rotations;
extern V1F cfg_anchor_generator_stride_matched_threshold;
extern V1F cfg_anchor_generator_stride_unmatched_threshold;
extern int cfg_max_number_of_points_per_voxel;
extern int cfg_max_number_of_voxels;
extern int cfg_nms_pre_max_size;  
extern int cfg_nms_post_max_size; 
extern float cfg_nms_iou_threshold;
extern float cfg_nms_score_threshold;
extern int cfg_num_class;
extern V1F cfg_post_center_limit_range;
extern std::vector<std::string> cfg_class_names;
// end config file item 

typedef struct{
 std::vector<float*> box_;
 std::vector<float*> cls_;
 std::vector<float*> dir_;
 int size_;
}DPU_DATA;

typedef struct{
 std::vector<float*> coors_;
 int size_;
 float scale_coors_;   // ?
}CPU_DATA;

typedef struct {
  V2F anchors;
  V2F anchors_bv;
}G_ANCHOR;

struct lex_cmp{
  bool operator ()( const std::array<float,4>& in1 , const std::array<float,4>& in2) const {
     if ( in1[0] != in2[0])  return in1[0] < in2[0];
     if ( in1[1] != in2[1])  return in1[1] < in2[1];
     return in1[2] > in2[2];
  }
};

using lex_queue = std::priority_queue<std::array<float,4>, std::vector<std::array<float,4> >, lex_cmp>; 

// =================== pre definition =====================

typedef Eigen::Tensor<int8_t, 3, Eigen::RowMajor>       VoxelsTensor;
typedef Eigen::TensorMap<VoxelsTensor>                  VoxelsTensorMap;

typedef Eigen::Tensor<float, 3, Eigen::RowMajor>        CoorsTensor;
typedef Eigen::TensorMap<CoorsTensor>                  CoorsTensorMap;

class preout_dict { 
  public:
      preout_dict( int8_t* dpu_in, int s0, int s1, int s2,
                    float* cpu_in, int t0, int t1, int t2 ) : // should be hwc :   12000 100  4
          voxelsData (dpu_in),
          s0_(s0), s1_(s1), s2_(s2),
          coorsData (cpu_in),
          t0_(t0), t1_(t1), t2_(t2)
      {
          memset(dpu_in, 0, s0*s1*s2);
          memset(cpu_in, 0, t0*t1*t2*sizeof(float));
          num_points.resize(s0);
          cur_size = 0;
      } 
      void SetSize(int s)   { 
         num_points.assign(s ,0);
         cur_size = s; 
      }
      void clear() { 
         memset(voxelsData, 0, s0_*s1_*s2_); 
         memset(coorsData, 0, t0_*t1_*t2_*sizeof(float)); 
      }  // 5000 100 4 
      int  GetSize()    { return cur_size; }
      V1I& GetNumPoints()   { return num_points; }
      VoxelsTensorMap GetVoxels() { return VoxelsTensorMap( voxelsData, s0_, s1_, s2_); }
      CoorsTensorMap GetCoors() { return CoorsTensorMap( coorsData, t0_, t1_, t2_); }
  private:
      int cur_size;
      V1I   num_points;
      int8_t* voxelsData;
      int s0_, s1_, s2_;
      float*  coorsData;
      int t0_, t1_, t2_;
};


struct MyV1I{
   MyV1I() : size_(0), maxsize_(0), buf(NULL){}
   void initialize(int size) {
      maxsize_ = size;
      if (buf == NULL) {
         buf = new int [size];
      } else {
         std::cerr <<"wrong usage for MyV1I \n";
         exit(-1);
      }
   }
   MyV1I(int size) : size_(0), maxsize_(size), buf(new int[size]){
   } 
   ~MyV1I(){ delete []buf; }
   void resize(int size) {
      if (maxsize_ < size ) {
          delete []buf;
          buf = new int[size];
          maxsize_ = size;
      }
      size_ = size;
   }
   int operator[](int idx) {
     CHECK_LT(idx, size_) << "index overflow...";
     return buf[idx];
   }
   int size()  { return size_; }
   int* data() { return buf; }
private:
   int size_;
   int maxsize_;
   int* buf;
};

// ===================================  help function  ========

template<typename T>
void printv2( std::string info, const T& inv) {
  std::cout <<"Debug ---------- " << info << "    size:" << inv.size() << std::endl;

  for (auto & v : inv )   {
    for (auto & v1 : v ) std::cout << " " << v1 << " ";
    std::cout << std::endl;
  }
}

template<typename T>
void printv1(const std::vector<T>& v, const std::string& vname ){
   std::cout <<"printv for " << vname << " size : " << v.size() << "\n";
   for(unsigned int i=0; i<v.size(); i++) {
     std::cout <<v[i] << " ";
   }
   std::cout <<"\n";
}

void printv( std::string info, const V2F& inv); 

std::string slurp(const char* filename);
std::string getEnvString(string envName, string defaultVal="");

template<typename T>
void myreadfile(T*conf, int size1, std::string filename);
template<typename T>
void mywritefile(T* src, int size1, std::string filename);

int getfloatfilelen(const std::string& file);
void clip_data(float* src_f, signed char*dst_c, int num, float scale);
void import_data(const std::string& filename, int8_t* dst_addr, float scale);

}}}


