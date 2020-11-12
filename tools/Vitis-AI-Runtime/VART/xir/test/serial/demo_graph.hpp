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

#pragma once

#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph_imp.hpp"
#include "xir/graph/serialize_v2.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_imp.hpp"

#include <iostream>
#include <string>
#include <vector>
using namespace std;

namespace xir {

class BuildGraphDemo {
 public:
  using InputOpsMap = std::map<std::string, std::vector<Op*>>;

 public:
  BuildGraphDemo() { g_ = make_shared<GraphImp>("demo_graph"); }

  ~BuildGraphDemo() = default;

 public:
  shared_ptr<GraphImp> build_graph_demo() {
    // add const op
    InputOpsMap const_iom;
    auto* const_op = add_const_op(const_iom);

    // add data op
    InputOpsMap data_iom;
    auto* data_op = add_data_op(data_iom);

    // add conv2d op
    InputOpsMap conv2d_iom = {
        {"weights", vector<Op*>{const_op}},
        {"input", vector<Op*>{data_op}},
    };
    auto* conv2d_op = add_conv2d_op(conv2d_iom);

    // add maxpool op
    InputOpsMap maxpool_iom = {
        {"input", vector<Op*>{conv2d_op}},
    };
    add_maxpool_op(maxpool_iom);

    return g_;
  }

 private:
  Op* add_const_op(const InputOpsMap& iom) {
    auto op_type = "const";
    auto op_name = "conv1_weights";
    auto attrs_ptr = Attrs::create();
    auto cur_op = g_->add_op(op_name, op_type, std::move(attrs_ptr), iom,
                             DataType{"INT8"});

    // todo: set cur_op's output tensor

    return cur_op;
  }

  Op* add_data_op(const InputOpsMap& iom) {
    auto op_type = "data";
    auto op_name = "conv1_data";
    auto attrs_ptr = Attrs::create();
    auto cur_op = g_->add_op(op_name, op_type, std::move(attrs_ptr), iom,
                             DataType{"INT8"});

    // todo: set cur_op's output tensor
    return cur_op;
  }

  Op* add_conv2d_op(const InputOpsMap& iom) {
    auto op_type = "conv2d";
    auto op_name = "conv2d_1";
    auto attrs_ptr = Attrs::create();
    attrs_ptr->set_attr("kernel_w", 7);
    attrs_ptr->set_attr("kernel_h", 8);
    attrs_ptr->set_attr("kernel_wd", 1);
    attrs_ptr->set_attr("kernel_hd", 2);
    attrs_ptr->set_attr("stride_w", 3);
    attrs_ptr->set_attr("stride_h", 4);
    attrs_ptr->set_attr("pad_l", 3);
    attrs_ptr->set_attr("pad_t", 4);
    attrs_ptr->set_attr("pad_r", 0);
    attrs_ptr->set_attr("pad_b", 1);
    attrs_ptr->set_attr("group", 1);

    auto cur_op = g_->add_op(op_name, op_type, std::move(attrs_ptr), iom,
                             DataType{"INT8"});

    // todo: set cur_op's output tensor
    return cur_op;
  }

  Op* add_maxpool_op(const InputOpsMap& iom) {
    auto op_type = "maxpool";
    auto op_name = "maxpool_1";
    auto attrs_ptr = Attrs::create();
    attrs_ptr->set_attr("kernel_w", 2);
    attrs_ptr->set_attr("kernel_h", 3);
    attrs_ptr->set_attr("kernel_wd", 1);
    attrs_ptr->set_attr("kernel_hd", 2);
    attrs_ptr->set_attr("stride_w", 3);
    attrs_ptr->set_attr("stride_h", 4);
    attrs_ptr->set_attr("pad_l", 3);
    attrs_ptr->set_attr("pad_t", 4);
    attrs_ptr->set_attr("pad_r", 0);
    attrs_ptr->set_attr("pad_b", 1);
    attrs_ptr->set_attr("ceil_mode", 2);
    attrs_ptr->set_attr("is_global", true);

    auto cur_op =
        g_->add_op(op_name, op_type, std::move(attrs_ptr), iom, {"INT8"});

    // todo: set cur_op's output tensor
    return cur_op;
  }

 private:
  shared_ptr<GraphImp> g_;
};

}  // namespace xir
