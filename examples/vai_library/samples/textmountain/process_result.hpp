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
#pragma once
#include <sys/stat.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/textmountain.hpp"

using namespace cv;
using namespace std;


static cv::Mat process_result( cv::Mat &img, 
           const vitis::ai::TextMountainResult &result, bool is_jpeg) {
 
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
  return img;
}

