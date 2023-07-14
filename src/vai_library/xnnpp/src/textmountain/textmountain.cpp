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

#include "vitis/ai/nnpp/textmountain.hpp"
#include "textmountain_postimp.hpp"

using namespace std;
namespace vitis {
namespace ai {

TextMountainPost::TextMountainPost() {}
TextMountainPost::~TextMountainPost() {}

std::unique_ptr<TextMountainPost> TextMountainPost::create(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    int batch_size,
    int& real_batch_size,
    float* scale_h,
    float* scale_w
) {
    return std::unique_ptr<TextMountainPost>(
       new TextMountainPostImp(input_tensors, output_tensors, 
              batch_size,  real_batch_size, scale_h, scale_w));
}

}  // namespace ai
}  // namespace vitis

