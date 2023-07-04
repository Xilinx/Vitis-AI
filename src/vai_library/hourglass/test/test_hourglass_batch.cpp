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
#include <vitis/ai/hourglass.hpp>

using namespace std;
using Result = vitis::ai::HourglassResult::PosePoint;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> ... <image_url>" << std::endl;
    abort();
  }

  auto det = vitis::ai::Hourglass::create(argv[1]);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

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

  vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 6},  {3, 6},  {3, 4}, {4, 5},
                                 {6, 7},   {7, 8},  {8, 9}, {7, 12},
                                 {12, 11}, {11, 10}, {7, 13}, {13, 14}, {14, 15}};

  // run model
  auto results = det->run(batch_images);

  // post process
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    cout << " batch index: " << batch_idx
         << "\timage name: " << batch_images_names[batch_idx] << endl;
    cout << "poses size: " << results[batch_idx].poses.size() << endl;
    for (size_t i = 0; i < results[batch_idx].poses.size(); ++i) {
      if (results[batch_idx].poses[i].type == 1) {
        cv::circle(batch_images[batch_idx],
                   results[batch_idx].poses[i].point, 5,
                   cv::Scalar(0, 255, 0), -1);
      }
    }
    for (size_t i = 0; i < limbSeq.size(); ++i) {
      Result a = results[batch_idx].poses[limbSeq[i][0]];
      Result b = results[batch_idx].poses[limbSeq[i][1]];
      if (a.type == 1 && b.type == 1) {
        cv::line(batch_images[batch_idx], a.point, b.point,
                 cv::Scalar(255, 0, 0), 3, 4);
      }
    }
    string tmp = batch_images_names[batch_idx].substr(
        0, batch_images_names[batch_idx].rfind("."));
    cv::imwrite("result_" + tmp + to_string(batch_idx) + ".jpg",
                batch_images[batch_idx]);
  }

  return 0;
}
