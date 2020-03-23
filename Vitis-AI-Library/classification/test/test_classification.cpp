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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/classification.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>" <<" <image_url>" << std::endl;
    abort();
  }
  string kernel = argv[1];
  auto det = vitis::ai::Classification::create(kernel);
  auto image = cv::imread(argv[2]);
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto res = det->run(image);

  for (const auto &r : res.scores) {
    cout << "index: " << r.index << " score " << r.score
         << " text: " << det->lookup(r.index) << endl;
  }
  return 0;
}
