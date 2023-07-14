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

namespace vitis { namespace ai {

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

}}

