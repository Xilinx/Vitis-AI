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
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/monodepth2.hpp>

using namespace std;
using namespace cv;
namespace fs=std::filesystem;

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
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name>  <test_file_list>  <result_dir>" 
              << "\n        model_name is monodepth2_pt " << std::endl;
    abort();
  }
  auto net = vitis::ai::Monodepth2::create(argv[1]);  (void)net;
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  std::vector<std::string> vlist;
  LoadListNames( std::string(argv[2]), vlist);

  auto ret = mkdir(argv[3], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[3] << std::endl;
    return -1;
  }
  
  int pos = 0;
  std::vector<float> dst(192*640);
  for(auto& f: vlist) {
     cv::Mat img = cv::imread(f);
     if (img.empty()) {
       std::cerr <<"bad file meet : " << f <<"\n";
       return -1;
     }
     auto res = net->run(img);
     float* p = res.mat.ptr<float>(0);
     std::stringstream ss;
     ss << argv[3] <<"/res_" << setw(4) << setfill('0') << pos <<".bin" ;
     mywritefile( p, 192*640, ss.str() );
     pos++;
  }
  return 0;
}

