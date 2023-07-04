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
#include <string>
#include <opencv2/opencv.hpp>
#include <thread>
#include <cstdint>
#include <stdio.h>
#include <unistd.h>
#include <vitis/ai/facerecog.hpp>

using namespace vitis::ai;
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    auto recog = FaceRecog::create("facerec_resnet20", true);
    if (!recog) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
    }   
    cout << "image url " << argv[1] << " " //
         << "inner_x " << argv[2] << " "   //
         << "inner_y " << argv[3] << " "   //
         << "inner_w " << argv[4] << " "   //
         << "inner_h " << argv[5] << " "   //
         << std::endl;

        cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
      cout << "cannot imread " << argv[1] << endl;
      exit(2);
    }


    auto result = recog->run_fixed(
        image, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));

    for (float feature : *result.feature) {
      cout << feature << " "; //
    }
    cout << std::endl;
}
