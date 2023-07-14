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
#include <vitis/ai/platedetect.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
using namespace vitis::ai;
int main(int argc, char *argv[]) {
  // bool preprocess = !(getenv("PRE") != nullptr);
  auto det = vitis::ai::PlateDetect::create(argv[1], true);
  if (!det) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;

  for (int i = 2; i < argc; i++) {
    auto image = cv::imread(argv[i]);
    if (image.empty()) {
      cerr << "cannot load " << argv[i] << endl;
      abort();
    }

    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);

    vitis::ai::PlateDetectResult res = det->run(img_resize);

    // 86 233 156 258
    cout << "res.box.score " << res.box.score << " "  //
         << "res.box.x " << res.box.x << " toint "
         << (int)(res.box.x * image.cols) << " "  //
         << "res.box.y " << res.box.y << " toint "
         << (int)(res.box.y * image.rows) << " "  //
         << "res.box.width " << res.box.width << " toint "
         << (int)(res.box.width * image.cols) << " "  //
         << "res.box.height " << res.box.height << " toint "
         << (int)(res.box.height * image.rows) << std::endl
         << "The real coordinate is: xx: "  //
         << "res.top_left :(" << res.top_left.x << " , " << res.top_left.y
         << ") "  //
         << "res.top_right :(" << res.top_right.x << " , " << res.top_right.y
         << ") "  //
         << "res.bottom_left :(" << res.bottom_left.x << " , "
         << res.bottom_left.y << ") "  //
         << "res.bottom_right :(" << res.bottom_right.x << " , "
         << res.bottom_right.y << ") "  //
         << std::endl;

    auto rect = cv::Rect{
        (int)(res.box.x * image.cols), (int)(res.box.y * image.rows),
        (int)(res.box.width * image.cols), (int)(res.box.height * image.rows)};
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255));

    cv::imshow("cout.jpg", image);
    cv::waitKey(0);
  }
  return 0;
}
