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

#include "UniLog/UniLog.hpp"
#include "xir/util/tool_function.hpp"

// helper function to add a conv
xir::Op* add_conv(std::string name, std::shared_ptr<xir::Graph> graph,
                  xir::Op* input);
// helper function to creat a graph
std::shared_ptr<xir::Graph> create_test_graph();

static uint32_t get_file_size(std::string file_name) {
  std::ifstream infile(file_name, std::ios::binary | std::ios::ate);
  UNI_LOG_CHECK(infile.is_open(), VAIEDIFF_BAD_FILE)
      << "Cannot open file " << file_name;
  return infile.tellg();
}

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto buf = std::vector<char>((size_t)get_file_size(__FILE__));
  CHECK(std::ifstream(__FILE__).read(&buf[0], buf.size()).good());
  UNI_LOG_INFO << "md5sum " << __FILE__ << " = "
               << xir::get_md5_of_buffer(&buf[0], buf.size());

  UNI_LOG_INFO << "md5sum " << __FILE__ << " = "
               << xir::get_md5_of_file(__FILE__);
  return 0;
}
