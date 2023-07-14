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
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>

using namespace std;
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main(int argc, char *argv[]) {
  off_t addr = stoul(argv[1]);
  auto width = stoi(argv[2]);
  auto height = stoi(argv[3]);
  long page_size = sysconf(_SC_PAGE_SIZE);
  //    off_t base = (addr / page_size) * page_size;  // page size alignment;
  size_t size = width * height * 3;
  size_t offset = size % page_size;
  size_t extra_size = size + offset;
  size_t map_size =
      (extra_size / page_size) * page_size +
      (extra_size % page_size == 0 ? 0 : page_size);  // page size alignment;
  auto fd = open("/dev/mem", O_RDWR | O_SYNC);
  CHECK_GT(fd, 0);
  auto capacity = extra_size;
  auto data =
      mmap(NULL, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr);
  CHECK_NE(data, MAP_FAILED) << "mmap failed";

  LOG(INFO) << "ptr->data<void>() " << data << " "
            << "width " << width << " "        //
            << "height " << height << " "      //
            << "map_size " << map_size << " "  //
            << endl;
  auto img = cv::Mat((int)height, (int)width, CV_8UC3);
  auto from = (unsigned char *)data;
  for (int i = 0; i < width * height * 3; ++i) {
    img.data[i] = from[i];
  }
  cv::imwrite("a.bmp", img);
  return 0;
}
