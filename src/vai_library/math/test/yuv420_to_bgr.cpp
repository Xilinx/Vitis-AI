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

#include "vitis/ai/math.hpp"
static void yuv420_to_bgr(int width, int height, const char *filename) {
  auto image_file_name = std::string(filename);
  auto fp = fopen(image_file_name.c_str(), "r");
  CHECK(fp != NULL);
  cv::Mat yuvImg_sp420;
  yuvImg_sp420.create(height * 3 / 2, width, CV_8UC1);
  CHECK_EQ(fread(yuvImg_sp420.data, height * width * 3 / 2, 1, fp), 1)
      << " width = " << width << " height = " << height
      << " filename = " << image_file_name;
  fclose(fp);
  cv::Mat bgrImg(height, width, CV_8UC3);
  vitis::ai::yuv2bgr(0, 0, width, height, yuvImg_sp420.data, width,
                     yuvImg_sp420.data + height * width, width, bgrImg.data);
  auto out_file =
      image_file_name.substr(0, image_file_name.size() - 7) + ".jpg";
  cv::imwrite(out_file, bgrImg);
  LOG(INFO) << "converting " << image_file_name << " to " << out_file;
}

int main(int argc, char *argv[]) {
  int width = std::stoi(argv[1]);
  int height = std::stoi(argv[2]);
  for (int i = 3; i < argc; ++i) {
    yuv420_to_bgr(width, height, argv[i]);
  }
  return 0;
}
