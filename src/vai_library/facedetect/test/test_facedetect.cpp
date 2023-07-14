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

using namespace std;

int main(int argc, char* argv[]) {
  bool preprocess = !(getenv("PRE") != nullptr);
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url> [<img_url> ...]"
              << std::endl;  //
    abort();
  }

  auto v = vitis::ai::FaceDetect::create(argv[1], preprocess);
  if (!v) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }   
  LOG(INFO) << "pre " << preprocess << " "
            << "v.get() " << (void*)v.get() << " "  //
            << std::endl;
  int width = v->getInputWidth();
  int height = v->getInputHeight();

  for (int i = 2; i < argc; i++) {
    cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ":"
         << "image " << argv[i] << " ";  //
    cv::Mat image = cv::imread(argv[i]);
    cv::Mat img_resize;

    cv::Mat canvas =
        cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar{0, 0, 0});
    auto ratio =
        std::min(static_cast<float>(width) / static_cast<float>(image.cols),
                 static_cast<float>(height) / static_cast<float>(image.rows));
    ratio = std::min(1.0f, ratio);
    auto new_w = image.cols * ratio;
    auto new_h = image.rows * ratio;
    auto new_size = cv::Size{(int)new_w, (int)new_h};
    cout << "new_w " << new_w << " "  //
         << "new_h " << new_h << " "  //
         << "ratio " << ratio << " "  //
         << std::endl;
    cv::resize(image, img_resize, new_size, 0, 0, cv::INTER_NEAREST);
    img_resize.copyTo(canvas(cv::Rect{cv::Point{0, 0}, new_size}));
    cv::imwrite("canvas.jpg", canvas);

    __TIC__(FACE_DET_TOTLE)
    auto result = v->run(canvas);
    __TOC__(FACE_DET_TOTLE)
    for (const auto& r : result.rects) {
      cout << " " << r.score << " "  //
           << r.x << " "             //
           << r.y << " "             //
           << r.width << " "         //
           << r.height << " "        //
           << endl;
      ;
      cv::rectangle(canvas,
                    cv::Rect{cv::Point(r.x * canvas.cols, r.y * canvas.rows),
                             cv::Size{(int)(r.width * canvas.cols),
                                      (int)(r.height * canvas.rows)}},
                    0xff);
    }
    cv::imwrite("out.jpg", canvas);
    cout << std::endl;
  }

  return 0;
}
