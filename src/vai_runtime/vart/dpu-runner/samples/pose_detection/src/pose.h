/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _14PT_HPP_
#define _14PT_HPP_

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "common.h"

using namespace std;
using namespace cv;
// JointPoint

namespace detect {

class GestureDetect {
 public:
  void Init(string& path);
  void Finalize();
  void Run(cv::Mat&);
  GestureDetect();
  ~GestureDetect();

 private:
  TensorShape inshapes[1];
  TensorShape outshapes[1];
  TensorShape fc_inshapes[1];
  TensorShape fc_outshapes[2];
  std::unique_ptr<xir::Graph> graph;
  std::unique_ptr<vart::Runner> pt_runner;
  std::unique_ptr<vart::Runner> fc_pt_runner;
};
}  // namespace detect

#endif
