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
#include <sstream>
#include "multitaskv3_onnx.hpp"

using namespace onnx_multitaskv3;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <imgurl>" << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);
  cv::Mat image = cv::imread(argv[2]);
  CHECK(!image.empty()) << "cannot read image from " << argv[2];

  auto det = OnnxMultiTaskv3::create(model_name);

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
  auto result = det->run(images);

  std::stringstream ss;
  for (int k = 0; k < (int)result.size(); k++) {
    std::cout << "batch-" << k << " \n";
    for (size_t i = 0; i < result[k].vehicle.size(); i++) {
      int xmin = result[k].vehicle[i].x * images[k].cols;
      int ymin = result[k].vehicle[i].y * images[k].rows;
      int width = result[k].vehicle[i].width * images[k].cols;
      int height = result[k].vehicle[i].height * images[k].rows;
      rectangle(images[k], Rect_<int>(xmin, ymin, width, height),
                Scalar(185, 181, 178), 2, 1, 0);
      cout << "label " << result[k].vehicle[i].label << " "          //
           << " score: " << result[k].vehicle[i].score << " xmin: "  //
           << xmin << " ymin: " << ymin << " width: " << width
           << " height: " << height << endl;
    }
    ss.str("");
    ss << "onnx_detection_" << k << ".jpg";
    cv::imwrite(ss.str(), images[k]);
    ss.str("");
    ss << "onnx_segmentation__" << k << ".jpg";
    cv::imwrite(ss.str(), result[k].segmentation);
    ss.str("");
    ss << "onnx_drivable_" << k << ".jpg";
    cv::imwrite(ss.str(), result[k].drivable);
    ss.str("");
    ss << "onnx_lane_" << k << ".jpg";
    cv::imwrite(ss.str(), result[k].lane);
    ss.str("");
    ss << "onnx_depth_" << k << ".jpg";
    cv::imwrite(ss.str(), result[k].depth);
  }
  return 0;
}

