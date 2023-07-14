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
#include <vitis/ai/pmg.hpp>

using namespace std;
using namespace cv;
std::vector<std::string> vlist;

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
    std::cerr << "usage :" << argv[0] << " <model_name> <test_list>  <result_file>"
              << "\n        model_name is pmg_pt " << std::endl;
    abort();
  }

  auto net = vitis::ai::PMG::create(argv[1]);
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  LoadListNames(argv[2],  vlist);

  ofstream Tout;
  Tout.open(argv[3], ios_base::out);
  if(!Tout) {
    cout<<"Can't open the file! " << argv[3] << "\n";
    return -1;
  }

  for(auto& it: vlist) {
     Mat img = cv::imread(it);
     if (img.empty()) {
       cerr << "cannot load " << it << endl;
       Tout << "0\n";
       continue;
     }
     auto result = net->run(img);
     // Tout << result.classidx  << "\n";
     Tout << result.classidx  << "\n";
  }

  Tout.close();
  return 0;
}

