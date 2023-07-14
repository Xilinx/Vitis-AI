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

#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/refinedet.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> ... <image_url>" << std::endl;
    abort();
  }

  auto det = vitis::ai::RefineDet::create(argv[1]);
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

  // run model
  auto results = det->run(batch_images);

  // post process
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    auto img = batch_images[batch_idx].clone();
    for (auto &box : results[batch_idx].bboxes) {
      float x = box.x * (img.cols);
      float y = box.y * (img.rows);
      int xmin = x;
      int ymin = y;
      int xmax = x + (box.width) * (img.cols);
      int ymax = y + (box.height) * (img.rows);
      float score = box.score;
      xmin = std::min(std::max(xmin, 0), img.cols);
      xmax = std::min(std::max(xmax, 0), img.cols);
      ymin = std::min(std::max(ymin, 0), img.rows);
      ymax = std::min(std::max(ymax, 0), img.rows);

      LOG(INFO) << "batch index: " << batch_idx << "\t" << xmin << "\t" << ymin
                << "\t" << xmax << "\t" << ymax << "\t" << score << "\n";
      // auto label = 2;
      // if (label == 1) {
      //   cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
      //                 cv::Scalar(0, 255, 0), 1, 1, 0);
      // } else if (label == 2) {
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255, 0, 0), 1, 1, 0);
      // } else if (label == 3) {
      //   cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
      //                 cv::Scalar(0, 0, 255), 1, 1, 0);
      // }
    }
    string tmp = batch_images_names[batch_idx].substr(
        0, batch_images_names[batch_idx].rfind("."));
    cv::imwrite("result_show_" + tmp + to_string(batch_idx) + ".jpg", img);
  }

  return 0;
}
