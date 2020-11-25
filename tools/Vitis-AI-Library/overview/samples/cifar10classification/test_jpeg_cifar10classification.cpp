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
#include <vitis/ai/demo.hpp>
#include <vitis/ai/cifar10classification.hpp>

#include "./process_result.hpp"
using namespace std;
int main(int argc, char *argv[]) {
  string model = argv[1];
  return vitis::ai::main_for_jpeg_demo(
      argc, argv, [model] { return vitis::ai::Cifar10Classification::create(model); },
      process_result, 2);
}

