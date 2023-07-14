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
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
void dump_op_weights(const std::string& reg_2_bin, const std::string& reg_3_bin,
                     const size_t n, const size_t w, const size_t h,
                     const size_t c, const uint64_t offset) {
  std::ifstream reg_2_file(reg_2_bin);
  std::ifstream reg_3_file(reg_3_bin);

  std::vector<char> reg_2_buf(std::istreambuf_iterator<char>(reg_2_file), {});
  std::vector<char> reg_3_buf(std::istreambuf_iterator<char>(reg_3_file), {});

  int hash_size = 4;
  int batch = hash_size / 2;
  auto size = n * h * w * c;
  auto buf = std::vector<char>(size);
  CHECK_EQ(buf.size(), size);
  auto step = h * w * c;

  auto i = 0ul;
  while (i < n) {
    auto d_batch = batch;
    if (i + d_batch > n) d_batch = n - i;
    auto buf_offset = i * step;
    auto reg_offset = offset + (i / hash_size) * batch * step;
    if (0)
      std::cout << "i " << i << " "                    //
                << "batch " << batch << " "            //
                << "reg  " << i % hash_size << " "     //
                << "buf_offset " << buf_offset << " "  //
                << "(i / hash_size) * batch * step "
                << (i / hash_size) * batch * step << " "  //
                << std::endl;

    if (i % hash_size) {
      for (auto l = 0ul; l < d_batch * step; l++) {
        buf[buf_offset + l] = reg_3_buf[reg_offset + l];
      }
    } else {
      for (auto l = 0ul; l < d_batch * step; l++) {
        buf[buf_offset + l] = reg_2_buf[reg_offset + l];
      }
    }

    i += d_batch;
  }
  CHECK(std::cout.write(&buf[0], size).good());
}

int main(int argc, char* argv[]) {
  auto reg_2_bin = std::string(argv[1]);
  auto reg_3_bin = std::string(argv[2]);
  auto n = std::stoi(std::string(argv[3]));
  auto h = std::stoi(std::string(argv[4]));
  auto w = std::stoi(std::string(argv[5]));
  auto c = std::stoi(std::string(argv[6]));
  auto offset = std::stoul(std::string(argv[7]));
  dump_op_weights(reg_2_bin, reg_3_bin, n, h, w, c, offset);
  return 0;
}
