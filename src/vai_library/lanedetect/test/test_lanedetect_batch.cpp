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
#include <opencv2/imgproc/imgproc_c.h>
#include <vitis/ai/lanedetect.hpp>

using namespace std;
using namespace cv;
using namespace vitis::ai;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> ... <image_url>" << std::endl;
    abort();
  }

  auto det = vitis::ai::RoadLine::create(argv[1]);
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
    cerr << "Cannot load images." << endl;
    abort();
  }

  // run model
  auto results = det->run(batch_images);

  // post process

  vector<int> color1 = {0, 255, 0, 0, 100, 255};
  vector<int> color2 = {0, 0, 255, 0, 100, 255};
  vector<int> color3 = {0, 0, 0, 255, 100, 255};

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    cout << " batch index: " << batch_idx
         << "\timage name: " << batch_images_names[batch_idx] << endl;
    for (auto &line : results[batch_idx].lines) {
      vector<Point> points_poly = line.points_cluster;
      int type = line.type < 5 ? line.type : 5;
      if (type == 2 && points_poly[0].x < batch_images[batch_idx].rows * 0.5)
        continue;
      cout << " points clouster size: " << points_poly.size() << endl;
      cv::polylines(batch_images[batch_idx], points_poly, false,
                    Scalar(color1[type], color2[type], color3[type]), 3, LINE_AA,
                    0);
    }
    string tmp = batch_images_names[batch_idx].substr(
        0, batch_images_names[batch_idx].rfind("."));
    cv::imwrite("result_" + tmp + to_string(batch_idx) + ".jpg",
                batch_images[batch_idx]);
  }
  return 0;
}
