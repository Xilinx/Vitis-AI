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

#include <fstream>
#include <iostream>
#include <queue>
#include <sys/stat.h>
#include <boost/algorithm/string.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/math.hpp>

#include "vitis/ai/nnpp/ocr.hpp"
#include "ocr_postimp.hpp"

using namespace std;
namespace vitis {
namespace ai {

OCRPost::OCRPost() {}
OCRPost::~OCRPost() {}

std::unique_ptr<OCRPost> OCRPost::create(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const std::string& cfgpath, 
    int batch_size,
    int& real_batch_size,
    std::vector<int>& target_h8,
    std::vector<int>& target_w8,
    std::vector<float>& ratioh,
    std::vector<float>& ratiow,
    std::vector<cv::Mat>& oriimg
) {
    return std::unique_ptr<OCRPost>(
       new OCRPostImp(input_tensors, output_tensors, 
              cfgpath, batch_size,  real_batch_size,
              target_h8, target_w8,
              ratioh, ratiow ,
              oriimg
    ));
}

}  // namespace ai
}  // namespace vitis

