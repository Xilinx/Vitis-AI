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
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/weak.hpp"
#include "xir/graph/graph.hpp"
#include "xir/graph/subgraph.hpp"

using namespace std;

namespace {
struct SubGraph_RIPV {
  SubGraph_RIPV(const xir::Subgraph* subg) {
    CHECK(subg->has_attr("reg_id_to_parameter_value"))
        << "failed to get DPU subgraph attr reg_id_to_parameter_value";
    ripv_map_ = subg->get_attr<std::map<std::string, std::vector<char>>>(
        "reg_id_to_parameter_value");
  }

  std::map<std::string, std::vector<char>> ripv_map_;
};

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto output_tensor = op->get_output_tensor();
    num_of_elements_ = output_tensor->get_element_num();
    dpu_ripv_ = nullptr;
    data_ptr_ = nullptr;
    data_vec_ = vector<char>();
    if (op->has_attr("data") &&
        !op->get_attr<std::vector<char>>("data").empty()) {
      data_vec_ = op->get_attr<vector<char>>("data");
      data_ptr_ = (int8_t*)&data_vec_[0];
    } else {
      auto root_subgraph = op->get_graph()->get_root_subgraph();
      auto subgraphs = root_subgraph->children_topological_sort();

      for (auto idx = 0; idx < (int)subgraphs.size(); ++idx) {
        if (subgraphs[idx]->has_op(op)) {
          auto device = subgraphs[idx]->get_attr<std::string>("device");
          if (device == "DPU") {
            CHECK(output_tensor->has_attr("reg_id"))
                << "failed to get op attr reg_id";
            auto reg_id = output_tensor->get_attr<int>("reg_id");
            CHECK(output_tensor->has_attr("ddr_addr"))
                << "failed to get op attr ddr_addr";
            auto ddr_addr = output_tensor->get_attr<int>("ddr_addr");

            dpu_ripv_ =
                vitis::ai::WeakStore<const xir::Subgraph*,
                                     SubGraph_RIPV>::create(subgraphs[idx],
                                                            subgraphs[idx]);
            std::vector<char>& ripv_data =
                dpu_ripv_->ripv_map_["REG_" + std::to_string(reg_id)];
            CHECK(!ripv_data.empty())
                << "reg_id_to_parameter_value data undefined";
            CHECK_LE(ddr_addr + num_of_elements_, ripv_data.size());
            data_ptr_ = (int8_t*)&ripv_data[ddr_addr];

            break;
          } else {
            CHECK(false) << "subgraph " << idx
                         << ": CPU subgraph has no data attr";
          }
        }
      }
    }
    CHECK(data_ptr_ != nullptr) << "cannot load valid const data";
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> result) {
    memcpy(result.data, data_ptr_, num_of_elements_);
    return 0;
  }

 private:
  std::shared_ptr<SubGraph_RIPV> dpu_ripv_;
  vector<char> data_vec_;
  int8_t* data_ptr_;
  int num_of_elements_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
