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
#include <iostream>
#include <string>
#include <xir/device_memory.hpp>
#include "parse_value.hpp"
using namespace std;
void dump_op_weights(void* buf, const size_t n, const size_t h, const size_t w,
                     const size_t c, const uint64_t reg_2_offset,
                     const uint64_t reg_3_offset) {
  /*
for (auto i = 0; i < n / hash_size; i++) {
  device_memory->download(&buf[(2 * i) * step * batch],
                          reg_2_offset + i * step * batch, step * batch);
  device_memory->download(&buf[(2 * i + 1) * step * batch],
                          reg_3_offset + i * step * batch, step * batch);
}
if (n % hash_size) {
  auto reg_2_batch = n % hash_size < batch ? n % hash_size : batch;
  device_memory->download(&buf[n / hash_size * step * hash_size],
                          reg_2_offset + n / hash_size * step * batch,
                          step * reg_2_batch);
  auto reg_3_batch = n % hash_size - reg_2_batch;
  if (reg_3_batch) {
    device_memory->download(
        &buf[n / hash_size * step * hash_size + step * batch],
        reg_3_offset + n / hash_size * step * batch, step * reg_3_batch);
  }
}
*/
}

int main(int argc, char* argv[]) {
  int n = 1;
  int h = 1;
  int w = 1;
  int c = 1;
  uint64_t reg_2_offset = 1;
  uint64_t reg_3_offset = 1;
  parse_value(std::string(argv[1]), n);
  parse_value(std::string(argv[2]), h);
  parse_value(std::string(argv[3]), w);
  parse_value(std::string(argv[4]), c);
  parse_value(std::string(argv[5]), reg_2_offset);
  parse_value(std::string(argv[6]), reg_3_offset);
  if (0)
    std::cout << "n " << n << " "                        //
              << "h " << h << " "                        //
              << "w " << w << " "                        //
              << "c " << c << " "                        //
              << "reg_2_offset " << reg_2_offset << " "  //
              << "reg_3_offset " << reg_3_offset << " "  //
              << std::endl;

  auto size = n * h * w * c;
  auto buf = std::vector<char>(size);
  CHECK_EQ(buf.size(), size);

  auto device_memory = xir::DeviceMemory::create((size_t)0ull);
  auto hash_size = 4;
  auto batch = hash_size / 2;
  auto step = h * w * c;
  int i = 0;
  while (i < n) {
    auto download_batch = batch;

    if (i + batch > n) {
      download_batch = n - i;
    }
    if (0)
      cout << "i " << i << " "
           << "download_batch " << download_batch << " "  //
           << endl;
    if (i % hash_size == 0) {
      if (0)
        cout << "reg_2 i * step " << i * step << " "                      //
             << "offset " << i / hash_size * batch * step << " "          //
             << "step * download_batch " << step * download_batch << " "  //

             << std::endl;
      device_memory->download(&buf[i * step],
                              reg_2_offset + i / hash_size * batch * step,
                              step * download_batch);
    } else {
      if (0)
        cout << "reg_3 i * step " << i * step << " "                      //
             << "offset " << i / hash_size * batch * step << " "          //
             << "step * download_batch " << step * download_batch << " "  //
             << std::endl;

      device_memory->download(&buf[i * step],
                              reg_3_offset + i / hash_size * batch * step,
                              step * download_batch);
    }
    i += download_batch;
  }
  if (0) {
    dump_op_weights(&buf[0], n, h, w, c, reg_2_offset, reg_3_offset);
  }

  CHECK(cout.write(&buf[0], size).good());

  return 0;
}
