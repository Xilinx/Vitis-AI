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
#include <glog/logging.h>
#include <xrt.h>

#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;

#include "parse_value.hpp"

int main(int argc, char* argv[]) {
  auto arg_addr = string(argv[1]);
  auto arg_width = string(argv[2]);
  auto arg_height = string(argv[3]);
  auto arg_file = string(argv[4]);

  unsigned long addr = 0;

  auto width = 0;             // stoi(argv[2]);
  auto height = 0;            // stoi(argv[3]);
  auto file = std::string();  // stoi(argv[3]);

  parse_value(arg_addr, addr);
  parse_value(arg_width, width);
  parse_value(arg_height, height);
  parse_value(arg_file, file);
  auto size = width * height * 3;
  auto deviceIndex = 0;
  auto handle = xclOpen(deviceIndex, NULL, XCL_INFO);
  auto img = cv::Mat((int)height, (int)width, CV_8UC3);
  auto flags = 0;
  auto ok = xclUnmgdPread(handle, flags, &img.data[0], size, addr);
  PCHECK(ok == 0) << "";
  cv::imwrite(file, img);
  xclClose(handle);
  return 0;
}
