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

#include <sys/stat.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/textmountain.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <model_name> <img_url>" << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }

  auto net = vitis::ai::TextMountain::create(argv[1]);
  if (!net) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto result = net->run(img);

  std::cout << "result num: " << result.res.size() << "\n";
  for(int i=0; i<(int)result.res.size(); i++) {
    std::cout << i << "   " << result.res[i].score << "\n"
              << "  " << result.res[i].box[0].x << " " <<  result.res[i].box[0].y << "\n"
              << "  " << result.res[i].box[1].x << " " <<  result.res[i].box[1].y << "\n"
              << "  " << result.res[i].box[2].x << " " <<  result.res[i].box[2].y << "\n"
              << "  " << result.res[i].box[3].x << " " <<  result.res[i].box[3].y << "\n" ;
    cv::line(img, result.res[i].box[0],  result.res[i].box[1], cv::Scalar(255, 0,   255)  );
    cv::line(img, result.res[i].box[1],  result.res[i].box[2], cv::Scalar(255, 0,   255)  );
    cv::line(img, result.res[i].box[2],  result.res[i].box[3], cv::Scalar(255, 0,   255)  );
    cv::line(img, result.res[i].box[3],  result.res[i].box[0], cv::Scalar(255, 0,   255)  );
  }
  cv::imwrite("textmountain_result.jpg", img);
  return 0;
}

