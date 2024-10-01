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

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/unet2d.hpp>

using namespace std;
using namespace cv;

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
  // for(int i=0; i<size1; i++)    Tout.write( (char*)conf+i*sizeof(T), sizeof(T));
  Tout.write( (char*)src, size1*sizeof(T));
}

void LoadListNames(const std::string& filename,  std::vector<std::string> &vlist)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if(!Tin)  {
     std::cout<<"Can't open the file " << filename << "\n";      exit(-1);
  }
  while( getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name> <image_list_file> <result_dir>" 
              << "\n        model_name is unet2d_tf2 " << std::endl;
    abort();
  }

  auto net = vitis::ai::Unet2D::create(argv[1]);

  vector<string> vlist;
  LoadListNames(argv[2], vlist);

  int batch = net->get_input_batch();
  int flen = getfloatfilelen(vlist[0]);
  std::vector<std::vector<float>> vf(batch, std::vector<float>(flen));
  std::vector<float*> vfp;
  vfp.reserve(batch);
  for(int i=0, j=0; i<(int)vlist.size(); ) {
    vfp.clear();
    for(j=0; j<batch && i<(int)vlist.size(); j++, i++) {
      myreadfile(vf[j].data(), flen, vlist[i]);
      vfp.emplace_back(vf[j].data());
    }
    auto ret = net->run(vfp, flen);
    for(int k=0; k<j; k++) {
      std::string filen = std::string(argv[3]) + vlist[i-j+k].substr(vlist[i-j+k].find_last_of('/') );
      mywritefile(ret[k].data.data(), ret[k].data.size(), filen);
    }
  }
 
  return 0;
}

