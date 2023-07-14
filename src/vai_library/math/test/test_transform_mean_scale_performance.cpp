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
#include <vitis/ai/profiling.hpp>

#include "../src/image_util.cpp"

using namespace vitis::ai;
int main(int argc, char* argv[]) {
  int max_src_size = std::stoi(argv[1]);
  std::cout << "begin test" << std::endl;
  std::vector<float> mean = {127, 114, 130};
  std::vector<float> scale = {1, 0.25, 0.0625};
  std::vector<int> scale_int;
  calc_scale(scale, scale_int);
  std::vector<uint8_t> src;
  std::vector<int8_t> res;
  for (int src_size = 100; src_size <= max_src_size; src_size += 100) {
    std::cout << "src_size " << src_size << std::endl;
    src.resize(3 * src_size);
    for (int i = 0; i < src_size; i++) {
      src[i * 3] = i;
      src[i * 3 + 1] = i;
      src[i * 3 + 2] = i;
    }
    res.resize(src.size());
    mean = {127, 114, 130};
    __TIC__(TRANSFORM_RGB_INT_MEAN);
    transform_rgb(src_size, 1, src.data(), res.data(), mean[0], scale[0],
                  mean[1], scale[1], mean[2], scale[2]);
    __TOC__(TRANSFORM_RGB_INT_MEAN);
    __TIC__(TRANSFORM_BGR_INT_MEAN);
    transform_bgr(src_size, 1, src.data(), res.data(), mean[0], scale[0],
                  mean[1], scale[1], mean[2], scale[2]);
    __TOC__(TRANSFORM_BGR_INT_MEAN);
    __TIC__(TRANSFORM_MEAN_SCALE_INT_MEAN_RGB);
    transform_mean_scale(src_size, 1, src.data(), res.data(), mean, scale_int,
                         false);
    __TOC__(TRANSFORM_MEAN_SCALE_INT_MEAN_RGB);
    __TIC__(TRANSFORM_MEAN_SCALE_INT_MEAN_BGR);
    transform_mean_scale(src_size, 1, src.data(), res.data(), mean, scale_int,
                         true);
    __TOC__(TRANSFORM_MEAN_SCALE_INT_MEAN_BGR);

    mean = {127.5, 114.3, 130.8};

    __TIC__(TRANSFORM_RGB_FLOAT_MEAN);
    transform_rgb(src_size, 1, src.data(), res.data(), mean[0], scale[0],
                  mean[1], scale[1], mean[2], scale[2]);
    __TOC__(TRANSFORM_RGB_FLOAT_MEAN);
    __TIC__(TRANSFORM_BGR_FLOAT_MEAN);
    transform_bgr(src_size, 1, src.data(), res.data(), mean[0], scale[0],
                  mean[1], scale[1], mean[2], scale[2]);
    __TOC__(TRANSFORM_BGR_FLOAT_MEAN);
    __TIC__(TRANSFORM_MEAN_SCALE_FLOAT_MEAN_RGB);
    transform_mean_scale(src_size, 1, src.data(), res.data(), mean, scale_int,
                         false);
    __TOC__(TRANSFORM_MEAN_SCALE_FLOAT_MEAN_RGB);

    __TIC__(TRANSFORM_MEAN_SCALE_FLOAT_MEAN_BGR);
    transform_mean_scale(src_size, 1, src.data(), res.data(), mean, scale_int,
                         true);
    __TOC__(TRANSFORM_MEAN_SCALE_FLOAT_MEAN_BGR);
    std::cout << "end test" << std::endl;
  }
  return 0;
}
