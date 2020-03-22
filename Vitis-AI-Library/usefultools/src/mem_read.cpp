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
#include "c++14.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <glog/logging.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>
using namespace std;

#define ERR_IF(cond, str)                                                      \
  if (cond) {                                                                  \
    cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "] "       \
         << endl;                                                              \
    perror(str);                                                               \
    exit(1);                                                                   \
  }

#include "parse_value.hpp"

int main(int argc, char *argv[]) {
  auto arg_addr = string(argv[1]);
  auto arg_size = string(argv[2]);
  unsigned long addr = 0;
  unsigned long size = 0;
  parse_value(arg_addr, addr);
  parse_value(arg_size, size);
  long page_size = sysconf(_SC_PAGE_SIZE);
  off_t offset = addr % page_size;
  off_t base = (addr / page_size) * page_size; // page size alignment;
  size_t extra_size = size + offset;
  size_t map_size =
      (extra_size / page_size) * page_size +
      (extra_size % page_size == 0 ? 0 : page_size); // page size alignment;

#ifdef __QNX__
  auto data = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                   MAP_PHYS | MAP_PRIVATE, NOFD, base);
#else
  auto fd = open("/dev/mem", O_RDWR | O_SYNC);
  CHECK_GT(fd, 0);
  auto data =
      mmap(NULL, extra_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
#endif
  fwrite(reinterpret_cast<const char *>(data) + offset, size, 1, stdout);
  fflush(stdout);
  munmap(data, map_size);
#ifndef __QNX__
  close(fd);
#endif
  return 0;
}
