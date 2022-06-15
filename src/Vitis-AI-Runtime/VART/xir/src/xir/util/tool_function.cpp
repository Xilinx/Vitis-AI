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

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <typeindex>

#include "../../../3rd-party/hash-library/md5.h"
#include "UniLog/UniLog.hpp"
#include "config.hpp"
#include "internal_util.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_def_factory_imp.hpp"
#include "xir/op/shape_inference.hpp"
#include "xir/util/data_type.hpp"

namespace xir {

const std::string get_md5_of_buffer(const void* buf, size_t size) {
  return MD5()(buf, size);
}

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
  auto strm = MD5();
  while (read_size) {
    strm.add(buffer, read_size);
    read_size = file.readsome(buffer, buffer_size);
  }
  delete[] buffer;
  file.close();
  auto ret = strm.getHash();
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

void register_customized_operator_definition(const std::string& name,
                                             const std::string& type) {
  UNI_LOG_WARNING
      << "The operator named " << name << ", type: " << type
      << ", is not defined in XIR. XIR creates the definition of this "
         "operator automatically. "
      << "You should specify the shape and "
         "the data_type of the output tensor of this operation by "
         "set_attr(\"shape\", std::vector<int>) and "
         "set_attr(\"data_type\", std::string)";
  auto new_operator =
      xir::OpDef(type)
          .add_input_arg(xir::OpArgDef{"input", OpArgDef::REPEATED,
                                       xir::DataType::FLOAT, ""})
          .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
              "shape", AttrDef::REQUIRED, 0,
              "`Datatype`: `vector<int>`\n\n"
              "The shape of the output tensor"))
          .add_attr(xir::AttrDefBuilder<std::string>::build(
              "data_type", AttrDef::REQUIRED,
              "`Datatype`: `string`\n\n"
              "The data type of the data of output feature maps, "
              "we use FLOAT32 as the default."))
          .set_annotation("This operator is not defined by XIR.")
          .set_shape_infer(xir::shape_infer_data);
  op_def_factory()->register_h(new_operator);
}

std::vector<float> get_float_vec_from_any(const xir::any& any) {
  auto type = std::type_index(any.type());
  auto f_vec = std::type_index(typeid(std::vector<float>));
  auto i_vec = std::type_index(typeid(std::vector<std::int32_t>));
  std::vector<float> fs;
  if (type == i_vec) {
    auto is = std::any_cast<std::vector<std::int32_t>>(any);
    for (auto i : is) fs.push_back(static_cast<float>(i));
  } else if (type == f_vec) {
    fs = std::any_cast<std::vector<float>>(any);
  } else {
    UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
        << "I cannot transform this xir::any to float.";
  }
  return fs;
}

bool TensorLexicographicalOrder(Tensor* a, Tensor* b) {
  return a->get_name() < b->get_name();
}
}  // namespace xir
