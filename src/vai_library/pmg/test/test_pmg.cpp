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
#include <vector>
#include <vitis/ai/pmg.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url>" << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }

  auto net = vitis::ai::PMG::create("pmg_pt");
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  auto result = net->run(img);

  std::cout << result.classidx<< "\n";

  return 0;
}

