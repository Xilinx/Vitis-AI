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

#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <vector>
#include <numeric>
#include <sys/stat.h>
#include <stdlib.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vitis/ai/brtseg.hpp>

using namespace cv;
using namespace std;
using namespace vitis::ai;

namespace fs=std::filesystem;

std::string out_dir;
std::string db_dir;
std::vector<std::string> vpath;

void accuracy_thread(Brtseg* det, int i, int t_n);

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage : " << argv[0] << " <model_name> <db_dir> <output_dir> [thread_num]" << std::endl;
    abort();
  }

  db_dir = std::string(argv[2]);
  out_dir = std::string(argv[3]);

  int t_n = 1; (void)t_n;
  if (argc==5) {
    t_n = atoi(argv[4]);
  }

  for(auto it=fs::directory_iterator(db_dir); it != fs::directory_iterator(); ++it) {
    if (fs::is_regular_file(*it)) {
       std::string str = it->path().string(); 
       str = str.substr(str.rfind("/")+1); 
       str = str.substr(0, str.rfind(".")); 
       vpath.emplace_back( str);
       // std::cout <<" string : " << vpath[vpath.size()-1] <<"\n"; 
    }
  }
  std::vector<std::thread> vth;
  std::vector< std::unique_ptr<Brtseg>> vssd;
  std::vector<std::string> out_dir_t(t_n);
  for(int i=0; i<t_n; i++) {
    vssd.emplace_back(vitis::ai::Brtseg::create(argv[1]));
    vth.emplace_back( std::thread( &accuracy_thread, vssd[i].get(), i , t_n ));
  }
  for(int i=0; i<t_n; i++) {
    vth[i].join();
  }
  return 0;
}

void accuracy_thread(Brtseg* det, int i, int t_n) 
{
  int all_size = (int)vpath.size();
  int split = all_size/t_n;
  int start = i*split;
  int end    = ((i != t_n-1) ? split : all_size-i*split);
  stringstream ss;
  for(int j=start; j<start+end; j++) {
    ss.str("");
    ss << db_dir << "/" << vpath[j] << ".png";
    // std::cout <<" vpath : " << vpath[j] << "    " << ss.str() <<"\n";
    std::string fn = ss.str();
    Mat img = cv::imread(fn);
    if (img.empty()) {
      cerr << "cannot load " << fn << endl;
      continue;
    }
    auto result = det->run(img);
    ss.str("");
    ss << out_dir <<"/" << vpath[j] << ".png";
    cv::imwrite(ss.str(), result.mat);
  }
}

