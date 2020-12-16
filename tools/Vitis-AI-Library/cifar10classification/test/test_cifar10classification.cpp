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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <algorithm>
#include <vitis/ai/cifar10classification.hpp>

using namespace cv;
using namespace std;

std::vector<std::string> obj = {
  "airplane", "automobile", "bird", "cat",  "deer",
  "dog", "frog", "horse", "ship", "truck"
};

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <modelname> <img_url>" << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto net = vitis::ai::Cifar10Classification::create(argv[1]);
  auto result = net->run(img);
  std::cout << "result: " << obj[result.classIdx] << "\n";
  return 0;
}
