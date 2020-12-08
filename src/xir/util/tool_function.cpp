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

#include "xir/util/tool_function.hpp"
#include <openssl/md5.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "UniLog/UniLog.hpp"
#include "config.hpp"
#include "internal_util.hpp"
#include "xir/util/data_type.hpp"

namespace xir {

const std::string get_md5_of_file(const std::string& filepath) {
  std::ifstream file(filepath.c_str(), std::ifstream::in);
  UNI_LOG_CHECK(!file.fail(), XIR_FILE_NOT_EXIST)
      << filepath << " doesn't exist.";
  UNI_LOG_DEBUG_INFO << "Read all data from " << filepath
                     << " to calculate the md5sum.";

  const std::uint32_t buffer_size = 1024;
  char* buffer;
  buffer = new char[buffer_size];

  auto read_size = file.readsome(buffer, buffer_size);
  unsigned char md5sum[MD5_DIGEST_LENGTH];
  MD5_CTX md5_context;
  MD5_Init(&md5_context);
  while (read_size) {
    MD5_Update(&md5_context, buffer, read_size);
    read_size = file.readsome(buffer, buffer_size);
  }
  MD5_Final(md5sum, &md5_context);
  std::stringstream ss;
  for (std::uint32_t idx = 0; idx < MD5_DIGEST_LENGTH; idx++) {
    ss << std::setfill('0') << std::setw(2) << std::hex
       << static_cast<std::uint32_t>(md5sum[idx]);
  }
  const std::string ret = ss.str();
  delete[] buffer;
  file.close();
  UNI_LOG_DEBUG_INFO << "md5sum(" << filepath << ") = " << ret << ".";
  return ret;
}

const std::string get_lib_name() {
  const auto ret =
      std::string{PROJECT_NAME} + "." + std::string{PROJECT_VERSION};
  return ret;
}

const std::string get_lib_id() {
  const auto ret = std::string{PROJECT_GIT_COMMIT_ID};
  return ret;
}

// name related
void add_prefix_helper(std::string& name, const std::string& prefix) {
  std::string prefix_inst = HEAD_DELIMITER + prefix + TAIL_DELIMITER;
  size_t insert_pos = 0;
  // name begin with  "__" will be hidden in serilization, so jump 2 position
  // before the prefix
  if (name.find_first_of("__") == 0) {
    insert_pos += 2;
  }
  name.insert(insert_pos, prefix_inst);
}

void add_suffix_helper(std::string& name, const std::string& suffix) {
  std::string suffix_inst = HEAD_DELIMITER + suffix + TAIL_DELIMITER;
  name += suffix_inst;
}

std::string remove_xfix(const std::string& name) {
  return *(extract_xfix(name).rbegin());
}

std::vector<std::string> extract_xfix(const std::string& name) {
  std::string head_delimiter = HEAD_DELIMITER;
  std::string tail_delimiter = TAIL_DELIMITER;
  std::string ret = name;
  std::vector<std::string> ret_vec;
  std::vector<std::size_t> head_pos_vec, tail_pos_vec;
  auto head_delimiter_pos = ret.find_first_of(head_delimiter);
  if (std::string::npos != head_delimiter_pos) {
    head_pos_vec.push_back(head_delimiter_pos);
  }
  while (head_pos_vec.size()) {
    auto current_head_delimiter_pos = *(head_pos_vec.rbegin());
    auto tail_delimiter_pos = ret.find(
        tail_delimiter, current_head_delimiter_pos + head_delimiter.size());
    if (std::string::npos != tail_delimiter_pos) {
      auto next_head_delimiter_pos = ret.find_first_of(
          head_delimiter, current_head_delimiter_pos + head_delimiter.size());
      // check if there a matching pair
      if ((std::string::npos == next_head_delimiter_pos) ||
          (next_head_delimiter_pos > tail_delimiter_pos)) {
        // if there's no next head delimiter or next head delimiter is after the
        // tail delimiter, remove one
        // collect the xfix
        ret_vec.push_back(
            ret.substr(current_head_delimiter_pos + head_delimiter.size(),
                       tail_delimiter_pos - current_head_delimiter_pos -
                           head_delimiter.size()));
        // remove the xfix and head_delimiter and tail_delimiter pair
        ret.erase(current_head_delimiter_pos, tail_delimiter_pos -
                                                  current_head_delimiter_pos +
                                                  tail_delimiter.size());
        // remove the current_head_delimiter_pos in head_pos_vec
        head_pos_vec.pop_back();
        // find the next head delimiter
        current_head_delimiter_pos =
            head_pos_vec.size() ? (*(head_pos_vec.rbegin())) : 0;
        next_head_delimiter_pos = ret.find_first_of(
            head_delimiter, current_head_delimiter_pos + head_delimiter.size());
        if (std::string::npos != next_head_delimiter_pos) {
          head_pos_vec.push_back(next_head_delimiter_pos);
        }
      } else {
        head_pos_vec.push_back(next_head_delimiter_pos);
        continue;
      }
    } else {
      // if there's no more tail_delimiter, break the loop
      break;
    }
  }
  ret_vec.push_back(ret);
  return ret_vec;
}

// math related
float xround(const float& data, const std::string& round_mode) {
  float ret;
  if ("STD_ROUND" == round_mode) {
    ret = std::round(data);
  } else if ("DPU_ROUND" == round_mode) {
    ret = internal::dpu_round_float(data);
  } else if ("PY3_ROUND" == round_mode) {
    ret = internal::py3_round_float(data);
  } else {
    UNI_LOG_FATAL(XIR_UNSUPPORTED_ROUND_MODE)
        << round_mode
        << " is not supported by xir now, if you require this mode, please "
           "contact us.";
  }
  return ret;
}

}  // namespace xir
