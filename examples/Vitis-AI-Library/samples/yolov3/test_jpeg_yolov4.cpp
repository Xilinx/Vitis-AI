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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/yolov3.hpp>
#include <glog/logging.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0] << " model_name  image_file_url " << endl;
    abort();
  }
  string image_file_name = argv[2];
  Mat img = cv::imread(image_file_name);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto yolo = vitis::ai::YOLOv3::create(argv[1], true);

  auto results = yolo->run(img);

  for (auto &box : results.bboxes) {
    int label = box.label;
    float xmin = box.x * img.cols;
    float ymin = box.y * img.rows;
    float xmax = xmin + box.width * img.cols;
    float ymax = ymin + box.height * img.rows;
    if (xmin < 0.) xmin = 1.;
    if (ymin < 0.) ymin = 1.;
    if (xmax > img.cols) xmax = img.cols;
    if (ymax > img.rows) ymax = img.rows;
    float confidence = box.score;
    LOG(INFO) << "RESULT: " << label << "\t" << xmin << "\t" << ymin
                          << "\t" << xmax << "\t" << ymax << "\t" << confidence
                          << "\n";
    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
              1, 0);
  }
  auto out_file =
      image_file_name.substr(0, image_file_name.size() - 4) + "_result.jpg";
  cv::imwrite(out_file, img);

  return 0;
}
