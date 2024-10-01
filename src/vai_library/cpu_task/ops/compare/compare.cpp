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
#include <iomanip>
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
#include "xir/cxir.h"
#include "xir/util/tool_function.hpp"

using namespace std;
DEF_ENV_PARAM(DEBUG_COMPARE, "0")
DEF_ENV_PARAM(COMPARE_QUIET, "0")
namespace {
std::string to_value(char v) {
  if (0) to_value(v);
  return std::to_string((int)v);
}

std::string to_value(unsigned char v) {
  if (0) to_value(v);
  return std::to_string((unsigned int)v);
}

std::string to_value(int v) {
  if (0) to_value(v);
  return std::to_string(v);
}

std::string to_value(float v) {
  if (0) to_value(v);
  ostringstream str;
  str << v;
  return str.str();
}

std::string to_hex(float v) {
  char* p = (char*)&v;
  ostringstream str;
  for (auto i = 0u; i < sizeof(float); ++i) {
    unsigned int v1 = p[i];
    str << std::hex << " " << v1;
  }
  return str.str();
}
template <typename T>
std::string to_binary_string(T v) {
  ostringstream str;
  auto p = (unsigned char*)&v;
  str << "0x";
  for (auto i = 0u; i < sizeof(v); ++i) {
    str << std::hex << std::setfill('0') << std::setw(2) << (unsigned int)p[i];
  }
  return str.str();
}

static std::string md5sum(const unsigned char* val, size_t size) {
  return xir::get_md5_of_buffer(val, size);
}

struct CompareOpImp : public vart::experimental::OpImpBase {
  CompareOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    // if (baseline_with_batch_.empty()) {
    //   if (op->has_attr("from_file")) {
    //     auto files = op->get_attr<std::vector<std::string>>("from_file");
    //     for (auto file : files) {
    //       auto baseline = std::vector<char>(vitis::ai::file_size(file));
    //       CHECK(std::ifstream(file)
    //                 .read((char*)&baseline[0], baseline.size())
    //                 .good())
    //           << "failed to read baseline from " << file;
    //       baseline_with_batch_.emplace_back(std::move(baseline));
    //     }
    //   }
    // }
    if (op->has_attr("md5sum")) {
      md5sum_ = op->get_attr<std::vector<std::string>>("md5sum");
    }
    log_limit_ = op->template get_attr<int>("log_limit");
    save_on_error_ = op->template get_attr<bool>("save_on_error");
    input_tensor_name_ =
        xir::remove_xfix(op->get_input_tensor("input", 0)->get_name());
    dump_directory_ = op->template get_attr<std::string>("dump_directory");
    input_op_ = op->get_input_op("input", 0);
    input_op_name_ = input_op_->get_name();
    input_op_type_ = input_op_->get_type();
    if (op->has_attr("baseline")) {
      baseline_with_batch_ =
          op->template get_attr<std::vector<std::vector<char>>>("baseline");
    }
    if (ENV_PARAM(DEBUG_COMPARE)) {
      auto bi = 0;
      for (const auto& b : baseline_with_batch_) {
        CHECK_LT(bi, (int)md5sum_.size());
        LOG(INFO) << "input_tensor_name_: " << input_tensor_name_
                  << "MD5: " << md5sum_[bi] << " "
                  << " base line " << bi << "; size = " << b.size();
        bi = bi + 1;
      }
      for (auto i = 0u; i < baseline_with_batch_.size(); ++i) {
        check_md5sum(i);
      }
    }
  };
  int calculate(vart::simple_tensor_buffer_t<int8_t> result,
                vart::simple_tensor_buffer_t<void> input) {
    auto batch_base = (size_t)attrs->get_attr<int>("__batch_base__");
    const auto& baseline = baseline_with_batch_[batch_base % md5sum_.size()];
    return calculate1<void>(result, input, baseline, batch_base);
  }

  template <typename T>
  int calculate1(vart::simple_tensor_buffer_t<int8_t> result,
                 vart::simple_tensor_buffer_t<T> input,
                 const std::vector<char>& baseline, size_t batch_base) {
    auto check_sum = md5sum((const unsigned char*)input.data, input.mem_size);
    auto expected = md5sum_[batch_base % md5sum_.size()];
    bool is_correct = (check_sum == expected);
    bool is_float = input_op_->get_output_tensor()->get_data_type().type ==
                    xir::DataType::FLOAT;
    bool has_baseline = !baseline.empty();
    if (!is_correct && is_float && has_baseline) {
      is_correct = true;
      CHECK_EQ(baseline.size(), input.mem_size);
      float* baseline_ptr = (float*)&baseline[0];
      float* input_ptr = (float*)input.data;
      auto size = input.mem_size / sizeof(float);
      auto error_count = 0;
      for (auto i = 0u; i < size; ++i) {
        bool is_same =
            fabs((input_ptr[i] - baseline_ptr[i]) / baseline_ptr[i]) <
            0.01f * 0.01f;  // accept 0.01% error
        is_correct = is_correct && is_same;
        if (!is_same) {
          LOG_IF(INFO, ENV_PARAM(DEBUG_COMPARE) && error_count < 10)
              << " error at " << i << " input_ptr[i] " << input_ptr[i] << " "
              << to_hex(input_ptr[i]) << " "  //
              << "baseline_ptr[i] " << baseline_ptr[i] << " "
              << to_hex(baseline_ptr[i])  //
              << " std::numeric_limits<float>::epsilon() "
              << std::numeric_limits<float>::epsilon() << " "  //
              << "input.tensor->get_name() " << input.tensor->get_name()
              << " "  //
              ;
        }
      }
      if (is_correct) {
        // pretend the result is correct, ignore md5sum check
        memcpy(&check_sum[0], &expected[0], 32);
      }
    }
    memcpy(&result.data[0], &check_sum[0], 32);
    memcpy(&result.data[32], &expected[0], 32);
    if (!is_correct && save_on_error_) {
      auto maybe_remove_trail_slah = [](const std::string& s) {
        if (s.back() == '/') {
          return s.substr(0, s.size() - 1);
        }
        return s;
      };
      std::string dir = maybe_remove_trail_slah(dump_directory_) + "/" +
                        std::to_string(batch_base);
      vitis::ai::create_parent_path(dir);
      auto filename =
          vitis::ai::to_valid_file_name(input_tensor_name_ + ".bin");
      auto fullname = dir + "/" + filename;
      ;
      CHECK(std::ofstream(fullname)
                .write((char*)input.data, input.mem_size)
                .good())
          << "failed to write: " << fullname;
      LOG_IF(INFO, !ENV_PARAM(COMPARE_QUIET))
          << "dump tensor " << input.tensor->get_name() << " on error."       //
          << " op: " << input_op_name_ << "; type=" << input_op_type_ << ";"  //
          << " actual_md5sum: " << check_sum                                  //
          << " expected_md5sum: " << expected                                 //
          << " filename=" << fullname                                         //
          << " data=" << (void*)input.data                                    //
          << " size=" << input.mem_size;
      ;
    } else {
      if (ENV_PARAM(DEBUG_COMPARE)) {
        LOG(INFO) << "congradulation!! compare OK: " << check_sum << " "
                  << input.tensor->get_name();
      }
    }
    return 0;
  }

 private:
  void check_md5sum(size_t i) {
    auto expected = md5sum_[i];
    if (expected == "00000000000000000000000000000000") {
      return;
    }
    if (baseline_with_batch_[i].empty()) {
      return;
    }
    auto actual = md5sum((const unsigned char*)&baseline_with_batch_[i][0],
                         baseline_with_batch_[i].size());
    CHECK_EQ(actual, expected) << " i=" << i;
  }

 private:
  std::vector<std::vector<char>> baseline_with_batch_;
  std::vector<std::string> md5sum_;
  int log_limit_;
  bool save_on_error_;
  std::string input_tensor_name_;
  const xir::Op* input_op_;
  std::string input_op_name_;
  std::string input_op_type_;
  std::string dump_directory_;
};

}  // namespace

DEF_XIR_OP_IMP(CompareOpImp)
