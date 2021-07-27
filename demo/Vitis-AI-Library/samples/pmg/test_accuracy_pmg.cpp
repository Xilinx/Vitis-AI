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

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/pmg.hpp>

using namespace std;
using namespace cv;
namespace fs=std::filesystem;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name> <test_dir>  <result_file>" 
              << "\n        model_name is pmg_pt " << std::endl;
    abort();
  }

  auto net = vitis::ai::PMG::create(argv[1]);

  std::string test_dir(argv[2]);  

  ofstream Tout;
  Tout.open(argv[3], ios_base::out);
  if(!Tout) {
    cout<<"Can't open the file! " << argv[3] << "\n";
    return -1;
  }

  std::vector<fs::path> vpath;
  std::vector<fs::path> vpath2;
  for(auto it=fs::directory_iterator(test_dir); it != fs::directory_iterator(); ++it) {
    if (fs::is_directory(*it)) {
       vpath.emplace_back(it->path() );
    }
  }
  std::sort(vpath.begin(), vpath.end());

  for(auto& it : vpath) {
    vpath2.clear();
    for(auto it2=fs::directory_iterator(it); it2 != fs::directory_iterator(); ++it2) {
      if (fs::is_regular_file(*it2)) {
         vpath2.emplace_back(it2->path() );
      }
    }
    std::sort(vpath2.begin(), vpath2.end());
    for(auto& it2: vpath2) {
       Mat img = cv::imread(it2.string());
       if (img.empty()) {
         cerr << "cannot load " << it2 << endl;
         Tout << "0\n";
         continue;
       }
       auto result = net->run(img);  
       // Tout << result.classidx  << "\n";
       Tout << result.classidx  << "\n";
    }   
  }

  Tout.close();
  return 0;
}
