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

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/movenet.hpp>

using namespace std;
using Result = vitis::ai::MovenetResult::PosePoint;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> ... <image_url>" << std::endl;
    abort();
  }

  auto det = vitis::ai::Movenet::create(argv[1]);

  // input process
  vector<cv::Mat> arg_input_images;
  vector<string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    arg_input_images.push_back(cv::imread(argv[i]));
    arg_input_images_names.push_back(argv[i]);
  }

  vector<cv::Mat> batch_images;
  vector<string> batch_images_names;
  auto batch = det->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  if (batch_images.empty()) {
    cerr << "cannot load images" << endl;
    abort();
  }
  vector<vector<int>> limbSeq = {{0, 1},   {0, 2},   {0, 3},   {0, 4},
                                 {0, 5},   {0, 6},   {5, 7},   {7, 9},
                                 {6, 8},   {8, 10},  {5, 11},  {6, 12},
                                 {11, 13}, {13, 15}, {12, 14}, {14, 16}};

  // run model
  auto results = det->run(batch_images);

  // post process
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    cout << " batch index: " << batch_idx
         << "\timage name: " << batch_images_names[batch_idx] << endl;
    cout << "poses size: " << results[batch_idx].poses.size() << endl;
    for (size_t i = 0; i < results[batch_idx].poses.size(); ++i) {
      cout << results[batch_idx].poses[i] << endl;
      if (results[batch_idx].poses[i].y > 0 &&
          results[batch_idx].poses[i].x > 0) {
        cv::putText(image, to_string(i), results[batch_idx].poses[i],
                    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 1,
                    0);
        cv::circle(image, results[batch_idx].poses[i], 5, cv::Scalar(0, 255, 0),
                   -1);
      }
    }
    for (size_t i = 0; i < limbSeq.size(); ++i) {
      auto a = results[batch_idx].poses[limbSeq[i][0]];
      auto b = results[batch_idx].poses[limbSeq[i][1]];
      if (a.x > 0 && b.x > 0) {
        cv::line(image, a, b, cv::Scalar(255, 0, 0), 3, 4);
      }
    }

    string tmp = batch_images_names[batch_idx].substr(
        0, batch_images_names[batch_idx].rfind("."));
    cv::imwrite("result_" + tmp + to_string(batch_idx) + ".jpg",
                batch_images[batch_idx]);
  }

  return 0;
}
