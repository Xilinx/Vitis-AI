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
#include "parse_value.hpp"
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

static unsigned long GetFileSize(const string &filename) {
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}
static unsigned long GetDefaultSize(const string &filename) {
  return GetFileSize(filename);
}

int main(int argc, char *argv[]) {
  // cout <<  __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__<< "] "
  //      << " " << options["addr"].as<off_t>()
  //      << " " << options["size"].as<unsigned long>()
  //      << " " << options["input"].as<vector<string> >().size()
  //      << endl;
  // exit(1);
  auto arg_addr = string(argv[1]);
  auto arg_file = string(argv[2]);
  unsigned long addr = 0;
  parse_value(arg_addr, addr);
  long page_size = sysconf(_SC_PAGE_SIZE);
  unsigned long offset = addr % page_size;
  unsigned long base = (addr / page_size) * page_size; // page size alignment;
  unsigned long size = GetDefaultSize(arg_file);
  unsigned long extra_size = size + offset;
  unsigned long map_size =
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
  vector<char> buf(page_size);
  FILE *fp = fopen(arg_file.c_str(), "rb");
  ERR_IF(fp == NULL, "open error");
  int c = 0;
  char *p = reinterpret_cast<char *>(data);
  p = p + offset;
  while ((c = fgetc(fp)) != EOF) {
    *p++ = (char)c;
  }

  munmap(data, map_size);
#ifndef __QNX__
  close(fd);
#endif
  fclose(fp);
  return 0;
}
