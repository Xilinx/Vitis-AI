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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/bcc.hpp>

using namespace std;
using namespace cv;

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
    std::cerr << "usage :" << argv[0] << " <model_name> <image_list_file>  <result_file>" 
              << "\n        model_name is bcc_pt " << std::endl;
    abort();
  }

  auto net = vitis::ai::BCC::create(argv[1]);

  vector<string> vlist;
  LoadListNames(argv[2], vlist);

  ofstream Tout;
  Tout.open(argv[3], ios_base::out);
  if(!Tout) {
    cout<<"Can't open the file! " << argv[4] << "\n";
    return -1;
  }

  for(auto &it: vlist) {
    Mat img = cv::imread(it);
    if (img.empty()) {
      cerr << "cannot load " << it << endl;
      abort();
    }
    auto result = net->run(img); 
    Tout << result.count  << "\n";
  }
  Tout.close();
  return 0;
}
