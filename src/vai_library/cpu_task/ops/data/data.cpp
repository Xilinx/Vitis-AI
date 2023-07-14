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

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "xir/util/tool_function.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
using namespace std;

namespace {
static std::string md5sum(const unsigned char* val, size_t size) {
  return xir::get_md5_of_buffer(val, size);
}

struct DataOpImp : public vart::experimental::OpImpBase {
  DataOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    if (op->has_attr("data")) {
      auto datas = op->get_attr<std::vector<std::vector<char>>>("data");
      for (auto& data : datas) {
        data_with_batch_.emplace_back(std::move(data));
      }
    }
    if (op->has_attr("md5sum")) {
      md5sum_ = op->get_attr<std::vector<std::string>>("md5sum");
    }
    CHECK(!md5sum_.empty());
    CHECK_EQ(md5sum_.size(), data_with_batch_.size());
    for (auto i = 0u; i < md5sum_.size(); ++i) {
      check_md5sum(i);
    }
  }
  int calculate(vart::simple_tensor_buffer_t<float> result) {
    auto batch_base = (size_t)attrs->get_attr<int>("__batch_base__");
    if (!data_with_batch_.empty()) {
      auto& data = data_with_batch_[batch_base % data_with_batch_.size()];
      CHECK_EQ(data.size(), result.mem_size) << "data size mismatch";
      memcpy((void*)result.data, (void*)&data[0], data.size());
    }
    return 0;
  }

 private:
  void check_md5sum(size_t i) {
    auto actual = md5sum((const unsigned char*)&data_with_batch_[i][0],
                         data_with_batch_[i].size());
    auto expected = md5sum_[i];
    CHECK_EQ(actual, expected) << " i=" << i;
  }

 private:
  std::vector<vector<char>> data_with_batch_;
  std::vector<std::string> md5sum_;
};
}  // namespace

DEF_XIR_OP_IMP(DataOpImp)
