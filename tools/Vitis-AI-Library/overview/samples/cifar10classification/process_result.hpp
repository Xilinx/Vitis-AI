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
#include <algorithm>
#include "vitis/ai/cifar10classification.hpp"

std::vector<std::string> obj = {
  "airplane", "automobile", "bird", "cat",  "deer",
  "dog", "frog", "horse", "ship", "truck"
};

using namespace cv;
using namespace std;

static cv::Mat process_result(
    cv::Mat &m1, const vitis::ai::Cifar10ClassificationResult &result,
    bool is_jpeg) {

    std::cout << "result: " << obj[result.classIdx] << "\n";
    return cv::Mat{};
}
