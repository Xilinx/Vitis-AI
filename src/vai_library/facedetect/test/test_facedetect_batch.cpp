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
#include <stdio.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/profiling.hpp>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  auto det = vitis::ai::FaceDetect::create(argv[1], true);
  if (!det) { // supress coverity complain
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

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  auto batch = det->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  int width = det->getInputWidth();
  int height = det->getInputHeight();

  std::vector<cv::Mat> new_images;
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    cv::Mat img_resize;
    cv::Mat canvas =
        cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar{0, 0, 0});
    auto ratio = std::min(static_cast<float>(width) /
                              static_cast<float>(batch_images[batch_idx].cols),
                          static_cast<float>(height) /
                              static_cast<float>(batch_images[batch_idx].rows));
    ratio = std::min(1.0f, ratio);
    auto new_w = batch_images[batch_idx].cols * ratio;
    auto new_h = batch_images[batch_idx].rows * ratio;
    auto new_size = cv::Size{(int)new_w, (int)new_h};
    std::cout << "new_w " << new_w << " "  //
              << "new_h " << new_h << " "  //
              << "ratio " << ratio << " "  //
              << std::endl;
    cv::resize(batch_images[batch_idx], img_resize, new_size, 0, 0,
               cv::INTER_NEAREST);
    img_resize.copyTo(canvas(cv::Rect{cv::Point{0, 0}, new_size}));
    cv::imwrite("resize_after_" + batch_images_names[batch_idx], canvas);
    new_images.push_back(canvas);
  }
  __TIC__(FACE_DET_TOTLE)
  auto results = det->run(new_images);
  __TOC__(FACE_DET_TOTLE)

  std::cout << std::endl;
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index: " << batch_idx << "   "                   //
              << "image_name: " << batch_images_names[batch_idx] << " "  //
              << std::endl;
    auto canvas = new_images[batch_idx];
    for (const auto &r : results[batch_idx].rects) {
      std::cout << " " << r.score << " "  //
                << r.x << " "             //
                << r.y << " "             //
                << r.width << " "         //
                << r.height << " "        //
                << std::endl;
      cv::rectangle(canvas,
                    cv::Rect{cv::Point(r.x * canvas.cols, r.y * canvas.rows),
                             cv::Size{(int)(r.width * canvas.cols),
                                      (int)(r.height * canvas.rows)}},
                    0xff);
    }
    std::cout << std::endl;
    cv::imwrite("out_" + batch_images_names[batch_idx], canvas);
  }

  return 0;
}
