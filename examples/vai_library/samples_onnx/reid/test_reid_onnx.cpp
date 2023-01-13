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

#include "reid_onnx.hpp"

using namespace cv;
double cosine_distance(Mat feat1, Mat feat2) { return 1 - feat1.dot(feat2); }
int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <pic1_url> <pic2_url> " << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);
  auto det = OnnxReid::create(model_name);
  auto batch = det->get_input_batch();

  Mat imgx = imread(argv[2]);
  if (imgx.empty()) {
    cerr << "can't load image! " << argv[2] << endl;
    return -1;
  }
  Mat imgy = imread(argv[3]);
  if (imgy.empty()) {
    cerr << "can't load image! " << argv[3] << endl;
    return -1;
  }

  std::vector<cv::Mat> imgxs(batch);
  std::vector<cv::Mat> imgys(batch);
  for (auto i = 0u; i < batch; ++i) {
    imgx.copyTo(imgxs[i]);
    imgy.copyTo(imgys[i]);
  }

  auto featxs = det->run(imgxs);
  auto featys = det->run(imgys);

  for (int k = 0; k < (int)featxs.size(); k++) {
    std::cout << "batch " << k << std::endl;
    double dismat = cosine_distance(featxs[k].feat, featys[k].feat);
    printf("dismat : %.3lf \n", dismat);
  }

  return 0;
}

