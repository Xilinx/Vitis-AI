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
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/medicalsegcell.hpp>

using namespace cv;
using namespace std;

void LoadListNames(const std::string& filename,
                   std::vector<std::string>& vlist) {
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if (!Tin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  while (getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << " usage: " << argv[0]
              << " <model_name> <img_list> <result_dir>" << std::endl;  //
    abort();
  }

  std::vector<std::string> vlist;
  LoadListNames(argv[2], vlist);

  auto ret = mkdir(argv[3], 0777);
  if (!(ret == 0 || (ret == -1 && EEXIST == errno))) {
    std::cout << "error occured when mkdir " << argv[3] << std::endl;
    return -1;
  }

  auto seg = vitis::ai::MedicalSegcell::create(argv[1]);
  if (!seg) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  Mat imgw;
  for (auto& it : vlist) {
    Mat img = cv::imread(it);
    if (img.empty()) {
      cerr << "cannot load " << it << endl;
      abort();
    }
    auto result = seg->run(img);
    cv::resize(result.segmentation, imgw, img.size(), 0, 0, cv::INTER_NEAREST);
    std::string filenamepart1 = it.substr(it.find_last_of('/') + 1);
    filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));
    filenamepart1 = std::string(argv[3]) + "/pred_" + filenamepart1 + ".png";
    cv::imwrite(filenamepart1, imgw);
  }
  return 0;
}
