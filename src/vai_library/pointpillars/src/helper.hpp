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
#include <vitis/ai/pointpillars.hpp>

using namespace std;

namespace vitis { namespace ai {

using V3F = std::vector<V2F>;
using V4F = std::vector<V3F>;
using V5F = std::vector<V4F>;
using V6F = std::vector<V5F>;
using V7F = std::vector<V6F>;

using V3I=std::vector<V2I>;

typedef struct{
 std::vector<int8_t*> box_;
 std::vector<int8_t*> cls_;
 std::vector<int8_t*> dir_;
 int size_;
 float scale_box_;
 float scale_cls_;
 float scale_dir_;
}DPU_DATA;

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

typedef Eigen::Tensor<int8_t, 3, Eigen::RowMajor>       PointPillarsScatterOutDpuTensor;
typedef Eigen::TensorMap<PointPillarsScatterOutDpuTensor> PointPillarsScatterOutDpuTensorMap;
typedef Eigen::Tensor<int8_t, 3, Eigen::RowMajor>       VoxelsTensor;
typedef Eigen::TensorMap<VoxelsTensor>                  VoxelsTensorMap;

class preout_dict 
{
    public:
        preout_dict( int8_t* dpu_in, int s0, int s1, int s2) : // should be hwc :   12000 100  4
            voxelsData (dpu_in)
        {
            memset(dpu_in, 0, s0*s1*s2);
            num_points.resize(s0);
            cur_size = 0;
            coorData.resize(s0 );
            voxelsShape[0] = s0;
            voxelsShape[1] = s1;
            voxelsShape[2] = s2;
        } 
        void SetSize(int s)   { 
           num_points.assign(s ,0);
           cur_size = s; 
        }
        void clear() { memset(voxelsData, 0, voxelsShape[0]*voxelsShape[1]*voxelsShape[2]); }  // 5000 100 4 
        int  GetSize()    { return cur_size; }
        V1I& GetNumPoints()   { return num_points; }
        VoxelsTensorMap GetVoxels() { return VoxelsTensorMap( voxelsData, voxelsShape[0], voxelsShape[1], voxelsShape[2] ); }
        std::vector< std::pair<int, int> > coorData;
    private:
        int cur_size;
        V1I   num_points;
        int8_t*     voxelsData;
        std::array<int, 3> voxelsShape;
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

}}


