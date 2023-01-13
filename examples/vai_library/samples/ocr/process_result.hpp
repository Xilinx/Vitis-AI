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
#pragma once
#include <sys/stat.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/ocr.hpp"

using namespace cv;
using namespace std;

Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(255, 255, 0), Scalar(0, 0, 255) };

static cv::Mat process_result( cv::Mat &img, 
           const vitis::ai::OCRResult &result, bool is_jpeg) {
 
   for(unsigned int i=0; i<result.words.size(); i++) {
      std::string str;
      for(auto& it: result.box[i]) {
         str+= std::to_string(it.x)+","+std::to_string(it.y)+",";
      }
      str+=result.words[i];
      std::cout << str <<"\n";

      cv::polylines(img, result.box[i], true, cv::Scalar(0, 0, 255), 2 );
      cv::putText(img, 
                  result.words[i], 
                  cv::Point(result.box[i][0].x+1, result.box[i][0].y+1 ), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                  cv::Scalar(0,0,0), 1);
      cv::putText(img, 
                  result.words[i], 
                  cv::Point(result.box[i][0].x, result.box[i][0].y), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                  cv::Scalar(0,255,255), 1);
   }
   return img;
}

