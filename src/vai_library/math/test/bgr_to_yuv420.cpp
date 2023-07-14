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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
static void bgr_to_yuv420(const char *filename) {
  auto image_file_name = std::string(filename);
  auto out_file =
      image_file_name.substr(0, image_file_name.size() - 4) + ".yuv420";
  auto image = cv::imread(filename);
  int w = image.cols;
  int h = image.rows;
  cv::Mat yuvImg_i420;
  cv::Mat rgbImg;
  cv::cvtColor(image, yuvImg_i420, CV_BGR2YUV_I420);
  cv::Mat yuvImg_sp420;
  yuvImg_sp420.create(h * 3 / 2, w, CV_8UC1);
  memcpy(yuvImg_sp420.data, yuvImg_i420.data, h * w);
  int c = w * h;
  for (auto i = 0; i < h * w / 4; ++i) {
    yuvImg_sp420.data[c++] = yuvImg_i420.data[h * w + h * w / 4 + i];
    yuvImg_sp420.data[c++] = yuvImg_i420.data[h * w + i];
  }
  auto fp = fopen(out_file.c_str(), "wb");
  if(fp) {
    fwrite(yuvImg_sp420.data, h, w * 3 / 2, fp);
    fclose(fp);
    LOG(INFO) << "converting " << image_file_name << " to " << out_file;
  } else {
    LOG(INFO) << "failed converting " << image_file_name;
  }
}

int main(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    bgr_to_yuv420(argv[i]);
  }
  return 0;
}
