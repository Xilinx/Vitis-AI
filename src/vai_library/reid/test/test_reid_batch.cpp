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
#include <vector>
#include <vitis/ai/reid.hpp>

using namespace std;
using namespace cv;

double cosine_distance(Mat feat1, Mat feat2) { return 1 - feat1.dot(feat2); }

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> ... <image_url> (at least two images)"
              << std::endl;
    abort();
  }

  auto det = vitis::ai::Reid::create(argv[1]);
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
  cout << "distant: " << endl;
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < results.size(); ++j) {
      Mat featx = results[i].feat;
      Mat featy = results[j].feat;
      double dismat = cosine_distance(featx, featy);
      printf("%.3lf \t", dismat);
    }
    cout << endl;
  }
  return 0;
}
