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
#include <fcntl.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>

using namespace std;
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main(int argc, char *argv[]) {
  auto file = std::string(argv[1]);
  auto width = stoi(argv[2]);
  auto height = stoi(argv[3]);
  auto file2 = std::string(argv[4]);
  auto img = cv::Mat((int)height, (int)width, CV_8UC3);
  CHECK(std::ifstream(file)
            .read(reinterpret_cast<char *>(&img.data[0]), width * height * 3)
            .good());
  cv::imwrite(file2, img);
  return 0;
}
