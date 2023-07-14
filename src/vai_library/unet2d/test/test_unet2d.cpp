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

#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/unet2d.hpp>

using namespace cv;
using namespace std;

int getfloatfilelen(const std::string& file)
{
  struct stat statbuf;
  if(stat(file.c_str(), &statbuf)!=0){
    std::cerr << " bad file stat " << file << std::endl;
    exit(-1);
  }
  return statbuf.st_size/4;
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
  Tout.write( (char*)src, size1*sizeof(T));
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <model> <img_url> [img_url]...." << std::endl;  //
    abort();
  }
  int len = getfloatfilelen(argv[2]);
  std::vector<std::vector<float>> vf(argc-2, std::vector<float>(len));
  std::vector<float*> vfp(argc-2);
  for(int i=2; i<argc; ++i) {
     myreadfile(vf[i-2].data(), len, argv[i]); 
     vfp[i-2] = vf[i-2].data();
  }
  
  auto net = vitis::ai::Unet2D::create(argv[1]);
  auto ret = net->run(vfp, len);

  int num = std::min(argc-2, (int)ret.size());
  for(int i=0; i<num; ++i) {
      std::string s(argv[i+2]);
      std::string filen = std::string("result_") + s.substr(s.find_last_of('/')+1);
      mywritefile(ret[i].data.data(), ret[i].data.size(), filen);
  }

  return 0;
}

