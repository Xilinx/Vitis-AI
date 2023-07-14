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
#include <fcntl.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>

using namespace std;
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main(int argc, char* argv[]) {
  auto file1 = std::string(argv[1]);
  auto file2 = std::string(argv[2]);
  auto width = stoi(argv[3]);
  auto height = stoi(argv[4]);
  cv::Mat img = cv::imread(file1);
  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(width, height), 0, 0);
  CHECK(std::ofstream(file2)
            .write(reinterpret_cast<char*>(&img.data[0]), width * height * 3)
            .good());
  return 0;
}
