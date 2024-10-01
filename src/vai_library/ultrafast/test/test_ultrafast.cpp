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
#include <sstream>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <vitis/ai/ultrafast.hpp>

using namespace cv;
using namespace std;

Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(255, 255, 0), Scalar(0, 0, 255) };

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url>" << std::endl; //
    abort();
  }

  Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }

  std::string name(argv[1]);

  std::string filenamepart1 = name.substr( name.find_last_of('/')+1 );
  filenamepart1 = filenamepart1.substr(0, filenamepart1.find_last_of('.'));

  auto det = vitis::ai::UltraFast::create("ultrafast_pt");
  if (!det) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  auto result = det->run(img);  (void)result;

  int iloop = 0;
  for(auto &lane: result.lanes) {
     std::cout <<"lane: " << iloop << "\n";
     for(auto &v: lane) {
        if(v.first >0) {
          cv::circle(img, cv::Point(v.first, v.second), 5, colors[iloop], -1);
        } 
        std::cout << "    ( " << v.first << ", " << v.second << " )\n";
     } 
     iloop++;
  }
  cv::imwrite(filenamepart1+"_result.jpg", img);

  return 0;
}

