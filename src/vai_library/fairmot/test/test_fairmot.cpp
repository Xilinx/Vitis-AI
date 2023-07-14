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
#include <vitis/ai/fairmot.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0] << "  modelname  image_file_url " << endl;
    abort();
  }
  auto image_file = string(argv[2]);
  Mat input_img = imread(image_file);
  if (input_img.empty()) {
    cerr << "can't load image! " << argv[2] << endl;
    return -1;
  }
  auto det = vitis::ai::FairMot::create(argv[1]);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto result = det->run(input_img);
  auto feats = result.feats;
  auto bboxes = result.bboxes;
  auto img = input_img.clone();
  for (auto i = 0u; i < bboxes.size(); ++i) {
    auto box = bboxes[i];
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

    LOG(INFO) << "RESULT " << box.label << " :\t" << xmin << "\t" << ymin
              << "\t" << xmax << "\t" << ymax << "\t" << score << "\n";
    LOG(INFO) << "feat size: " << feats[i].size()
              << " First 5 digits: " << feats[i].data[0] + 0.0f << " "
              << feats[i].data[1] + 0.0f << " " << feats[i].data[2] + 0.0f
              << " " << feats[i].data[3] + 0.0f << " "
              << feats[i].data[4] + 0.0f << endl;
    cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  cv::Scalar(0, 255, 0), 1, 1, 0);
  }
  auto out = image_file.substr(0, image_file.size() - 4) + "_out.jpg";
  LOG(INFO) << "write result to " << out;
  cv::imwrite(out, img);
  return 0;
}
