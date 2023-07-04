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
#include <opencv2/opencv.hpp>
#include <vitis/ai/retinaface.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url>" << std::endl;
    abort();
  }

  string kernel = argv[1];
  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl; abort(); }

  auto retinaface = vitis::ai::RetinaFace::create(kernel, true);
  if (!retinaface) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  int width = retinaface->getInputWidth();
  int height = retinaface->getInputHeight();

  std::cout << "input width " << width << " "    //
            << "input height " << height << " "  //
            << std::endl;

  int valid_width = 640;
  int valid_height = 360;
  std::cout << "valid_width " << valid_width << " "    //
            << "valid_height " << valid_height << " "  //
            << std::endl;

  float scale = 360.0 / img.rows;
  if (img.cols * scale > 640) {
    scale = 640.0 / img.cols;
  }

  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(img.cols * scale, img.rows * scale),
             0, 0, cv::INTER_LINEAR);
  std::cout << "resize width " << img_resize.cols<< " "    //
            << " height " << img_resize.rows << std::endl;

  cv::Mat input_image(384, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat roi = input_image(cv::Range{0, img_resize.rows}, cv::Range{0, img_resize.cols});

  img_resize.copyTo(roi);

  auto results = retinaface->run(input_image);
  cout << "results.size:" << results.bboxes.size();
  for (auto i = 0u; i < results.bboxes.size(); ++i) {
    auto &box = results.bboxes[i];
    auto &landmark = results.landmarks[i];
    float fxmin = box.x * input_image.cols;
    float fymin = box.y * input_image.rows;
    float fxmax = fxmin + box.width * input_image.cols;
    float fymax = fymin + box.height * input_image.rows;
    float confidence = box.score;

    int xmin = round(fxmin * 100.0) / 100.0;
    int ymin = round(fymin * 100.0) / 100.0;
    int xmax = round(fxmax * 100.0) / 100.0;
    int ymax = round(fymax * 100.0) / 100.0;

    xmin = std::min(std::max(xmin, 0), input_image.cols);
    xmax = std::min(std::max(xmax, 0), input_image.cols);
    ymin = std::min(std::max(ymin, 0), input_image.rows);
    ymax = std::min(std::max(ymax, 0), input_image.rows);

    cout << "RESULT: " << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    rectangle(input_image, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
                1, 0);
    cout << "landmark : ";
    for (auto j = 0; j < 5; ++j) {
      auto px = landmark[j].first * input_image.cols;
      auto py = landmark[j].second * input_image.rows;
      cv::circle(input_image, Point(px, py), 1, Scalar(255, 255, 0), 1);
      cout << " [" << px << ", " << py << "], ";
    }
    cout << endl;
  }

  cv::imwrite("out.jpg", input_image);
  return 0;
}
