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

#include <sstream>
#include <fstream>
#include <sys/stat.h>

#include "./helper.hpp"
using namespace std;

namespace vitis { namespace ai { namespace pp {

// begin config file item 
V1F cfg_voxel_size{0.16, 0.16, 4};
V1F cfg_point_cloud_range{0, -39.68, -3, 69.12, 39.68, 1};
V1I cfg_layer_strides{2, 2, 2};
V2F cfg_anchor_generator_stride_sizes{
 {1.6, 3.9, 1.56},
 {0.6, 1.76, 1.73},
 {0.6, 0.8, 1.73}    };
V2F cfg_anchor_generator_stride_strides{
 {0.32, 0.32, 0.0},
 {0.32, 0.32, 0.0},
 {0.32, 0.32, 0.0}  };
V2F cfg_anchor_generator_stride_offsets{
 {0.16, -39.52, -1.78},
 {0.16, -39.52, -1.465},
 {0.16, -39.52, -1.465}  };
V2F cfg_anchor_generator_stride_rotations{
 {0, 1.57},
 {0, 1.57},
 {0, 1.57} };
V1F cfg_anchor_generator_stride_matched_threshold{0.6, 0.5, 0.5};
V1F cfg_anchor_generator_stride_unmatched_threshold{0.45, 0.35, 0.35};
int cfg_max_number_of_points_per_voxel{100};
int cfg_max_number_of_voxels{12000};
int cfg_nms_pre_max_size{1000};  
int cfg_nms_post_max_size{300}; 
float cfg_nms_iou_threshold{0.5};
float cfg_nms_score_threshold{0.5};
int cfg_num_class{3};
V1F cfg_post_center_limit_range{0, -39.68, -5, 69.12, 39.68, 5};
std::vector<std::string> cfg_class_names{ "Car", "Cyclist", "Pedestrian"};
// end config file item 

std::string slurp(const char* filename) {
  std::ifstream in;
  std::stringstream sstr;
  try {
    in.open(filename, std::ifstream::in);
    sstr << in.rdbuf();
    in.close();
    if (sstr.str().empty()) {
      throw -1;
    }
  } catch (...) {
     std::cerr << "failed to open file " << filename <<"\n";
  }
  return sstr.str();
}

template<typename T>
void myreadfile(T* dest, int size1, std::string filename)
{ 
  ifstream Tin;  
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {     
     cout<<"Can't open the file! " << filename << std::endl; 
     return; 
  }
  // for(int i=0; i<size1; i++) { Tin.read( (char*)dest+i*4, 4); }
  Tin.read( (char*)dest, size1*sizeof(T));
}

template void myreadfile(float*dest, int size1, std::string filename);
template void myreadfile(int*dest, int size1, std::string filename);

template<typename T>
void mywritefile(T* src, int size1, std::string filename)
{ 
  ofstream Tout;  
  Tout.open(filename, ios_base::out|ios_base::binary);
  if(!Tout)  {   
     cout<<"Can't open the file! " << filename << "\n";
     return;
  }
  // for(int i=0; i<size1; i++)    Tout.write( (char*)conf+i*sizeof(T), sizeof(T)); 
  Tout.write( (char*)src, size1*sizeof(T));
}

template void mywritefile(int8_t* src, int size1, std::string filename);

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
}


std::string getEnvString(string envName, string defaultVal)
{
    char* val = getenv(envName.c_str());
    if (val) {
        return val;
    } else {
        return defaultVal;
    }
}

void clip_data(float* src_f, signed char*dst_c, int num, float scale)
{
  for(int i=0; i<num; i++) {
    if (src_f[i] > 127*scale ) {
       dst_c[i]=127;
    } else if (src_f[i] < -128*scale ){
       dst_c[i]=-128;  // std::cout <<" smaller for arm_loc!\n";
    } else {
      dst_c[i] = (signed char)(src_f[i]/scale);
    }
  }
}

void import_data(const std::string& filename, int8_t* dst_addr, float scale) {
    int len = getfloatfilelen( filename);
    float* fbuf=new float[ len ];
    myreadfile( fbuf, len, filename);
    clip_data(fbuf, dst_addr, len,   scale >1.0 ? 1.0/(1.0*scale) : scale ) ;
    delete []fbuf;
}

}}}

