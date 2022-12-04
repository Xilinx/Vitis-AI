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

#include <fstream>
#include "segmentation_onnx.hpp"

using namespace onnx_segmentation;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <image_name> " << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);
  auto image_name = std::string(argv[2]);
  cv::Mat image = imread(image_name);
  CHECK(!image.empty()) << "cannot read image from " << image_name;

  auto det = OnnxSegmentation::create(model_name);
  auto batch = det->get_input_batch();
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    image.copyTo(images[i]);
  }

  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;

  cout << "start running " << endl;
  auto res = det->run(images);
  for (auto i = 0u; i < res.size(); ++i) {
    auto name = std::to_string(i) + "_result_" + image_name;
    cout << "saving " << name << endl;
    cv::imwrite(name, res[i].segmentation);
  }

  return 0;
}

