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
 *
 * Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights
 * Reserved.
 */
#include "vart/zero_copy_helper.hpp"

#include <glog/logging.h>

#include <algorithm>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_ZERO_COPY_HELPER, "0")
namespace vart {

static bool has_reg_id_to_content_type(const xir ::Subgraph* subgraph) {
  return subgraph->has_attr("reg_id_to_context_type") ||
         subgraph->has_attr("reg_id_to_context_type_v2");
}

static std::map<std::string, std::string> get_reg_id_to_content_type(
    const xir ::Subgraph* subgraph) {
  auto ret = std::map<std::string, std::string>{};
  if (subgraph->has_attr("reg_id_to_context_type_v2")) {
    ret = subgraph->get_attr<std::map<std::string, std::string>>(
        "reg_id_to_context_type_v2");
  } else if (subgraph->has_attr("reg_id_to_context_type")) {
    ret = subgraph->get_attr<std::map<std::string, std::string>>(
        "reg_id_to_context_type");
  } else {
    CHECK(false)
        << "cannot find reg_id_to_context_type or reg_id_to_context_type_v2";
  }
  return ret;
}

static int get_reg_index(const std::string& reg_id) {
  auto index = reg_id.find("REG_");
  CHECK(index == 0) << "reg id is not support! reg_id = " << reg_id;

  auto str = reg_id.substr(4);
  for (size_t i = 0; i < str.size(); i++) {
    CHECK(str[i] >= '0' && str[i] <= '9')
        << "reg id is not support! reg_id = " << reg_id;
  }

  auto ret = std::stoi(str);
  CHECK_LT(ret, MAX_REG_ID_SIZE)
      << "reg id exceeds max supported " << MAX_REG_ID_SIZE;

  return ret;
}

static int get_max_reg_index(const std::map<std::string, std::string> reg_ids) {
  int max = 0;
  for (auto& reg : reg_ids) {
    max = std::max(max, get_reg_index(reg.first));
  }
  return max;
}

struct op_output_tensor_ddr {
  size_t reg_id;
  size_t ddr_addr;
  size_t location;
};

static bool belong_to_subgraph(const xir::Op* op,
                               const xir::Subgraph* subgraph) {
  auto ops = subgraph->get_ops();
  return ops.find(op) != ops.end();
}

static op_output_tensor_ddr get_op_output_tensor_ddr(
    const xir::Op* op, const xir::Subgraph* subgraph) {
  auto tensor = op->get_output_tensor();
  if (op->get_type() == "download" && belong_to_subgraph(op, subgraph)) {
    auto input_ops = op->get_input_ops("input");
    CHECK_EQ(input_ops.size(), 1u)
        << "There must be only one pre_op for download op";
    tensor = input_ops[0]->get_output_tensor();
  } else if (!tensor->has_attr("reg_id") ||
             (!belong_to_subgraph(op, subgraph))) {
    auto fanout_ops = op->get_fanout_ops();
    auto subgraph_ops1 = subgraph->get_ops();
    auto subgraph_ops =
        std::vector<const xir::Op*>(subgraph_ops1.begin(), subgraph_ops1.end());
    auto ops = std::vector<const xir::Op*>();
    std::sort(fanout_ops.begin(), fanout_ops.end());
    std::sort(subgraph_ops.begin(), subgraph_ops.end());
    std::set_intersection(fanout_ops.begin(), fanout_ops.end(),
                          subgraph_ops.begin(), subgraph_ops.end(),
                          std::back_inserter(ops));
    CHECK_EQ(ops.size(), 1u)
        << "illegal xmodel. op:" << op->get_name() << "  has no ddr info";
    auto upload_op = ops.front();
    CHECK_EQ(upload_op->get_type(), "upload")
        << "illegal xmodel. op:" << op->get_name() << "  has no ddr info";
    /*    auto up_next_op = upload_op->get_fanout_ops();
        CHECK_EQ(up_next_op.size(), 1u)
            << "illegal xmodel. op:" << op->get_name() << "  has no ddr info";
        auto next_op = up_next_op.front();
    */
    tensor = upload_op->get_output_tensor();
  }
  CHECK(tensor->has_attr("reg_id")) << "op_name " << op->get_name();
  CHECK(tensor->has_attr("ddr_addr")) << "op_name " << op->get_name();
  CHECK(tensor->has_attr("location")) << "op_name " << op->get_name();
  auto reg_id = (size_t)tensor->template get_attr<int>("reg_id");
  auto ddr_addr = (size_t)tensor->template get_attr<int>("ddr_addr");
  auto location = (size_t)tensor->template get_attr<int>("location");

  return op_output_tensor_ddr{reg_id, ddr_addr, location};
}

static std::vector<op_output_tensor_ddr> get_tensor_ddr_info(
    const xir::Subgraph* subgraph,
    const std::vector<const xir::Tensor*>& tensors) {
  auto ret = std::vector<op_output_tensor_ddr>();
  for (auto xir_tensor : tensors) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_ZERO_COPY_HELPER) >= 2)
        << "searchign for tensor ddr info for : tensor: "
        << xir_tensor->get_name();
    auto op = xir_tensor->get_producer();
    CHECK(op != nullptr) << "cannot find tensor's producer: "
                         << xir_tensor->get_name();
    auto tensor_ddr = get_op_output_tensor_ddr(op, subgraph);
    ret.push_back(tensor_ddr);
  }
  return ret;
}

static bool has_reg_id_in_tensors(
    int reg_id, const std::vector<op_output_tensor_ddr>& info) {
  auto ret = false;
  for (auto& x : info) {
    if ((int)x.reg_id == reg_id) {
      ret = true;
      break;
    }
  }
  return ret;
}

static std::string to_string(const std::vector<op_output_tensor_ddr> infos) {
  std::ostringstream str;
  str << "[";
  for (auto info : infos) {
    str << "{info.ddr_addr " << info.ddr_addr << ","  //
        << "info.location " << info.location << ","
        << "info.reg_id " << info.reg_id  //
        << "}";
  }
  str << "]";
  return str.str();
}
static std::string more_debug_info(
    const xir::Subgraph* subgraph,
    const std::vector<op_output_tensor_ddr> inputs,
    const std::vector<op_output_tensor_ddr> outputs) {
  std::ostringstream str;
  str << "subgraph: " << subgraph->get_name() << "inputs " << to_string(inputs)
      << " "                                       //
      << "outputs " << to_string(outputs) << " ";  //
  return str.str();
}

static std::string to_string(const vart::reg_basic_info_t& info) {
  std::ostringstream str;
  str << "\treg_id = " << info.reg_id << ";\n";
  str << "\ttype = " << to_string(info.type) << ";\n";
  str << "\tsize = " << info.size << ";\n";
  return str.str();
}
static std::string to_string(const std::vector<vart::reg_basic_info_t>& infos) {
  std::ostringstream str;
  str << "\n{";
  for (auto info : infos) {
    str << to_string(info);
  }
  str << "},\n";
  return str.str();
}

std::vector<vart::reg_basic_info_t> extract_reg_info_from_subgraph(
    const xir::Subgraph* subgraph_) {
  auto ret = std::vector<reg_basic_info_t>();
  if (!(has_reg_id_to_content_type(subgraph_) &&
        subgraph_->has_attr("reg_id_to_size"))) {
    return {};
  }
  auto reg_id_to_context_type = get_reg_id_to_content_type(subgraph_);
  auto reg_id_to_size =
      subgraph_->get_attr<std::map<std::string, int>>("reg_id_to_size");

  auto input_ddr_info =
      get_tensor_ddr_info(subgraph_, subgraph_->get_sorted_input_tensors());
  auto output_ddr_info =
      get_tensor_ddr_info(subgraph_, subgraph_->get_sorted_output_tensors());

  ret.resize(get_max_reg_index(reg_id_to_context_type) + 1);
  for (auto& reg : reg_id_to_context_type) {
    auto reg_id = reg.first;
    auto reg_type = reg.second;
    auto reg_size = reg_id_to_size[reg_id];
    auto reg_id_int = get_reg_index(reg_id);
    CHECK(ret[reg_id_int].type == reg_type_t::INVALID);
    auto& reg_info = ret[reg_id_int];
    reg_info.reg_id = reg_id_int;
    {  // begin initialize the reg_info_t
      if (reg_type == "CONST") {
        reg_info.type = reg_type_t::CONST;
      } else if (reg_type == "WORKSPACE") {
        reg_info.type = reg_type_t::DATA_GLOBAL;
      } else if (reg_type == "DATA") {
        reg_info.type = reg_type_t::DATA_LOCAL;
      } else if (reg_type == "INTERFACE") {
        auto is_input_reg_id =
            has_reg_id_in_tensors(reg_id_int, input_ddr_info);
        auto is_output_reg_id =
            has_reg_id_in_tensors(reg_id_int, output_ddr_info);
        if (is_input_reg_id && is_output_reg_id) {
          reg_info.type = reg_type_t::DATA_LOCAL;
        } else if (is_input_reg_id && (!is_output_reg_id)) {
          reg_info.type = reg_type_t::DATA_LOCAL_INPUT;
        } else if ((!is_input_reg_id) && is_output_reg_id) {
          reg_info.type = reg_type_t::DATA_LOCAL_OUTPUT;
        } else {
          CHECK(false) << "invalid type: "
                       << more_debug_info(subgraph_, input_ddr_info,
                                          output_ddr_info)
                       << "reg_type " << reg_type << " "                  //
                       << "reg_id_int " << reg_id_int << " "              //
                       << "is_input_reg_id " << is_input_reg_id << " "    //
                       << "is_output_reg_id " << is_output_reg_id << " "  //
                       << std::endl;
        }
      }
      reg_info.size = reg_size;
    }
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_ZERO_COPY_HELPER) >= 1) << to_string(ret);
  return ret;
}

static int get_tensor_buffer_size(
    const xir::Subgraph* subgraph,
    const std::vector<const xir::Tensor*>& tensors) {
  auto tensor_ddr_info = get_tensor_ddr_info(subgraph, tensors);
  auto reg_info = extract_reg_info_from_subgraph(subgraph);
  CHECK(!tensor_ddr_info.empty());
  auto first_reg = tensor_ddr_info.front();
  auto first_reg_id = first_reg.reg_id;
  auto all_tensors_have_same_reg_id =
      std::all_of(tensor_ddr_info.begin() + 1, tensor_ddr_info.end(),
                  [first_reg_id](const struct op_output_tensor_ddr& info) {
                    // info.
                    return info.reg_id == first_reg_id;
                  });
  if (!all_tensors_have_same_reg_id) {
    LOG(WARNING) << "the model does not support zero-copy, because there are "
                    "more than 1 ddr address  space for input/output()";
    return -1;
  }
  auto& the_reg_info = reg_info[first_reg_id];
  CHECK(the_reg_info.type == reg_type_t::DATA_LOCAL_INPUT ||
        the_reg_info.type == reg_type_t::DATA_LOCAL_OUTPUT)
      << "the model must support io-split";
  return (int)the_reg_info.size;
}

int get_input_buffer_size(const xir::Subgraph* subgraph) {
  auto input_tensors = subgraph->get_sorted_input_tensors();
  LOG_IF(INFO, ENV_PARAM(DEBUG_ZERO_COPY_HELPER) >= 2)
      << "searching for input tensors " << input_tensors.size()
      << " subgraph=" << subgraph->get_name();
  return get_tensor_buffer_size(subgraph, input_tensors);
}
std::vector<size_t> get_input_offset(const xir::Subgraph* subgraph) {
  // be careful about the output is not stable ordered.
  auto input_tensors = subgraph->get_sorted_input_tensors();
  auto ddr_info = get_tensor_ddr_info(subgraph, input_tensors);
  auto ret = std::vector<size_t>();
  ret.reserve(input_tensors.size());
  for (auto info : ddr_info) {
    ret.emplace_back(info.ddr_addr);
  }
  return ret;
}

int get_output_buffer_size(const xir::Subgraph* subgraph) {
  auto output_tensors = subgraph->get_sorted_output_tensors();
  LOG_IF(INFO, ENV_PARAM(DEBUG_ZERO_COPY_HELPER) >= 2)
      << "searching for output tensors " << output_tensors.size()
      << " subgraph=" << subgraph->get_name();
  return get_tensor_buffer_size(subgraph, output_tensors);
}
std::vector<size_t> get_output_offset(const xir::Subgraph* subgraph) {
  auto output_tensors = subgraph->get_sorted_output_tensors();
  auto ddr_info = get_tensor_ddr_info(subgraph, output_tensors);
  auto ret = std::vector<size_t>();
  ret.reserve(output_tensors.size());
  for (auto info : ddr_info) {
    ret.emplace_back(info.ddr_addr);
  }
  return ret;
}

}  // namespace vart
