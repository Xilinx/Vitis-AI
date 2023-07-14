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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

#include "./ssr.hpp"
int main(int argc, char* argv[]) {
  string model_name = argv[1];
  Mat img = cv::imread(argv[2]);
  {
    auto ssr = vitis::ai::SSR::create(model_name);
    if (!ssr) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
    }
    vector<Mat> imgs;
    for(size_t i = 0; i < ssr->get_input_batch(); ++i)
      imgs.push_back(img);
    if (1) {
      ssr->run(imgs);
      auto result = ssr->get_result();
      int c = 0;
      for (auto& r : result) {
        // imshow(std::string("result ") + std::to_string(c), result[c]);
        imwrite(std::string("result_ssr_") + std::to_string(c) + ".jpg", r);
        c++;
      }
      if (c != 0) {
        waitKey(0);
      }
    }
  }
  LOG(INFO) << "BYEBYE";
  return 0;
}

