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

#include <sstream>
#include "movenet_onnx.hpp"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <pic1_url> " << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);

  auto det = OnnxMovenet::create(argv[1]);
  vector<vector<int>> limbSeq = {{0, 1},   {0, 2},   {0, 3},   {0, 4},
                                 {0, 5},   {0, 6},   {5, 7},   {7, 9},
                                 {6, 8},   {8, 10},  {5, 11},  {6, 12},
                                 {11, 13}, {13, 15}, {12, 14}, {14, 16}};

  auto batch = det->get_input_batch();
  auto image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cerr << "cannot load " << argv[2] << std::endl;
    abort();
  }
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    image.copyTo(images[i]);
  }

  auto results = det->run(images);
  for (int k = 0; k < (int)results.size(); k++) {
    std::cout << "batch " << k << "\n";
    for (size_t i = 0; i < results[k].poses.size(); ++i) {
      cout << results[k].poses[i] << endl;
      if (results[k].poses[i].y > 0 && results[k].poses[i].x > 0) {
        cv::putText(images[k], to_string(i), results[k].poses[i],
                    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 1,
                    0);
        cv::circle(images[k], results[k].poses[i], 5, cv::Scalar(0, 255, 0),
                   -1);
      }
    }
    for (size_t i = 0; i < limbSeq.size(); ++i) {
      auto a = results[k].poses[limbSeq[i][0]];
      auto b = results[k].poses[limbSeq[i][1]];
      if (a.x > 0 && b.x > 0) {
        cv::line(images[k], a, b, cv::Scalar(255, 0, 0), 3, 4);
      }
    }
    stringstream ss;
    ss << "sample_movenet_result_" << k << ".jpg";
    cv::imwrite(ss.str(), images[k]);
  }
  return 0;
}

