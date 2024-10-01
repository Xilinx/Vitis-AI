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
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <vitis/ai/cflownet.hpp>

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
void myreadfile(T* dest, const std::string& filename)
{
  ifstream Tin;
  Tin.open(filename, ios_base::in|ios_base::binary);
  if(!Tin)  {
     cout<<"Can't open the file! " << filename << std::endl;
     abort();
  }
  int size1 = getfloatfilelen(filename);
  Tin.read( (char*)dest, size1*sizeof(float));
}

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
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <binfile_url>" << std::endl;  //
    abort();
  }

  std::vector<float> vf(128*128);
  myreadfile(vf.data(), std::string(argv[1]));
  
  auto net = vitis::ai::Cflownet::create("cflownet_pt");
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto result = net->run(vf.data() ); (void)result;
  mywritefile(result.data.data(), 128*128, "./cflownet_result.bin");

  return 0;
}

