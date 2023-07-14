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
    std::cerr << "usage :" << argv[0]
              << " <model_name> <image_url> ... <image_url> " << std::endl;
    abort();
  }

  auto retinaface = vitis::ai::RetinaFace::create(argv[1], true);
  if (!retinaface) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  } 
  std::vector<cv::Mat> arg_input_images;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  int width = retinaface->getInputWidth();
  int height = retinaface->getInputHeight();
  auto batch = retinaface->get_input_batch();
  std::vector<cv::Mat> batch_images;
  std::vector<cv::Mat> batch_input_images(batch);
  std::vector<std::string> batch_images_names;

  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  std::vector<float> scales(batch);
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    auto rows = batch_images[batch_idx].rows;
    auto cols = batch_images[batch_idx].cols;
    float scale = 360.0 / rows; 
    if (cols * scale > 640) {
      scale = 640.0 / cols;
    }
    scales[batch_idx] = scale;
    cv::Mat img_resize;
    cv::resize(batch_images[batch_idx], img_resize, cv::Size(cols * scale, rows * scale),
               0, 0, cv::INTER_LINEAR);
    cv::Mat input_image(384, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat roi = input_image(cv::Range{0, img_resize.rows}, cv::Range{0, img_resize.cols});
    img_resize.copyTo(roi);
    input_image.copyTo(batch_input_images[batch_idx]);
  }
  
  auto results = retinaface->run(batch_input_images);

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index " << batch_idx << " "                     //
              << "image_name " << batch_images_names[batch_idx] << " "  //
              << std::endl;
    for (auto &box : results[batch_idx].bboxes) {
      float fxmin = box.x * width / scales[batch_idx];  
      float fymin = box.y * height / scales[batch_idx]; 
      float fwidth = box.width * width / scales[batch_idx]; 
      float fheight = box.height * height / scales[batch_idx];
      float fxmax = fxmin + fwidth; 
      float fymax = fymin + fheight;
      float confidence = box.score;

      int xmin = round(fxmin * 100.0) / 100.0;
      int ymin = round(fymin * 100.0) / 100.0;
      int xmax = round(fxmax * 100.0) / 100.0;
      int ymax = round(fymax * 100.0) / 100.0;
      //int w = round(fwidth * 100.0) / 100.0;
      //int h = round(fheight * 100.0) / 100.0;

      xmin = std::min(std::max(xmin, 0), batch_images[batch_idx].cols);
      xmax = std::min(std::max(xmax, 0), batch_images[batch_idx].cols);
      ymin = std::min(std::max(ymin, 0), batch_images[batch_idx].rows);
      ymax = std::min(std::max(ymax, 0), batch_images[batch_idx].rows);

      std::cout << "RESULT-" << batch_idx << ": " << xmin
                << "\t" << ymin << "\t" << xmax << "\t" << ymax << "\t"
                << confidence << "\n";
    }
    std::cout << std::endl;
  }

  return 0;
}
