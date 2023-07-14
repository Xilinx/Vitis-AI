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
#include "dpu_kernel.hpp"

#include <glog/logging.h>

#include <UniLog/UniLog.hpp>
#include <algorithm>
#include <fstream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/trace.hpp>
#include <vitis/ai/weak.hpp>
#include <vitis/ai/xxd.hpp>

DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(XLNX_ENABLE_DEBUG_MODE, "0");
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_PARAMTER, "0");
namespace vart {
namespace dpu {
// ret[id] id is not meaningful
static std::vector<DpuReg> create_workspaces(const xir::Subgraph& subgraph);

static std::shared_ptr<GraphHolder> create_graph_holder(
    const std::string& filename, const std::string& kernel) {
  auto ret = vitis::ai::WeakStore<std::string, GraphHolder>::create(filename,
                                                                    filename);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "filename " << filename << " "           //
      << "kernel " << kernel << " "               //
      << "ret.get() " << (void*)ret.get() << " "  //
      << std::endl;
  return ret;
}

DpuKernel::DpuKernel(const std::string& filename, const std::string& kernel)
    : graph_holder_{create_graph_holder(filename, kernel)},
      subgraph_{graph_holder_->get_subgraph(kernel)},
      workspace_regs_(create_workspaces(*subgraph_)),
      super_layer_subgraph_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "create dpu kernel. "
      << "graph " << subgraph_->get_graph()->get_name() << ";"
      << "sub graph " << subgraph_->get_name() << " @"
      << (const void*)subgraph_;
}

DpuKernel::DpuKernel(const xir::Subgraph& sg,
                     xir::Attrs* attrs /* not used yet */)
    : graph_holder_{nullptr},
      subgraph_{&sg},
      workspace_regs_(create_workspaces(*subgraph_)),
      super_layer_subgraph_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "create dpu kernel. "
      << "graph " << subgraph_->get_graph()->get_name() << ";"
      << "sub graph " << subgraph_->get_name() << " @"
      << (const void*)subgraph_;
}

DpuKernel::~DpuKernel() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "kernel destoryed. "
      << "graph " << subgraph_->get_graph()->get_name() << ";"
      << "sub graph " << subgraph_->get_name() << " @"
      << (const void*)subgraph_;
}

void DpuKernel::initialize() {
  my_load_parameter();

  vitis::ai::trace::add_subgraph(subgraph_);

  if (ENV_PARAM(XLNX_ENABLE_DEBUG_MODE) == 0) {
    my_load_release_code();
  } else {
    my_load_debug_code();
  }
}

void DpuKernel::my_load_parameter() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "loading parameter for " << subgraph_->get_graph()->get_name();
  // CHECK(subgraph_->has_attr("reg_id_to_parameter_value"))
  //     << "subgraph name:" << subgraph_->get_name();
  auto reg_id_to_parameter_value = std::map<std::string, std::vector<char>>();
  if (subgraph_->has_attr("reg_id_to_parameter_value")) {
    reg_id_to_parameter_value =
        subgraph_->get_attr<std::map<std::string, std::vector<char>>>(
            "reg_id_to_parameter_value");
  }

  UNI_LOG_CHECK(subgraph_->has_attr("reg_id_to_context_type"),
                VART_XMODEL_ERROR);
  auto reg_id_to_context_type =
      subgraph_->get_attr<std::map<std::string, std::string>>(
          "reg_id_to_context_type");
  std::vector<DpuReg> parameters;
  size_t total = 0;
  for (auto& reg : reg_id_to_context_type) {
    auto reg_id = reg.first;
    auto reg_type = reg.second;
    if (reg_type != "CONST") {
      continue;
    }
    auto it_value = reg_id_to_parameter_value.find(reg_id);
    UNI_LOG_CHECK(it_value != reg_id_to_parameter_value.end(),
                  VART_XMODEL_ERROR)
        << "cannot find CONST REG values:"
        << " subgraph name= " << subgraph_->get_name()
        << " reg_id = " << reg_id;

    total = total + it_value->second.size();
    if (ENV_PARAM(XLNX_ENABLE_DUMP_PARAMTER)) {
      auto filename = std::string("") + reg_id + ".bin";
      auto mode =
          std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
      CHECK(std::ofstream(filename, mode)
                .write(&it_value->second[0], it_value->second.size())
                .good())
          << " faild to dump code to " << filename;
      LOG(INFO) << "dump parameter to " << filename;
    }
    parameters.emplace_back(reg_id, vart::dpu::RegType::XCONST,
                            std::move(it_value->second));
  }
  // CHECK_GT(total, 0u) << "no parameter loaded.";
  if (total != 0u) {
    load_parameter(parameters);
  }
}
void DpuKernel::my_load_release_code() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "loading release code for " << subgraph_->get_graph()->get_name();
  UNI_LOG_CHECK(subgraph_->has_attr("mc_code"), VART_XMODEL_ERROR)
      << "subgraph_->get_name() " << subgraph_->get_name() << " "  //
      << "attrs: " << subgraph_->get_attrs()->debug_info();
  auto& mc_code = subgraph_->get_attr<std::vector<char>>("mc_code");
  load_code(DpuReg{"REG_CODE", RegType::CODE, mc_code});
  super_layer_subgraph_.emplace_back(subgraph_);
}

static bool is_tilling_subgraph(const xir::Subgraph* subgraph) {
  if (subgraph->has_attr("children_topological_sort") &&
      subgraph->has_attr("tiling")) {
    return true;
  }
  return false;
}

void DpuKernel::my_load_debug_code() {
  UNI_LOG_CHECK(subgraph_->has_attr("children_topological_sort"),
                VART_XMODEL_ERROR)
      << "subgraph_->get_name() " << subgraph_->get_name() << " "  //
      << "attrs: " << subgraph_->get_attrs()->debug_info();
  UNI_LOG_CHECK(!subgraph_->is_leaf(), VART_XMODEL_ERROR)
      << "subgraph_->get_name() " << subgraph_->get_name() << " "  //
      << "attrs: " << subgraph_->get_attrs()->debug_info();

  auto children = subgraph_->get_children();
  std::vector<std::string> child_order =
      subgraph_->get_attr<std::vector<std::string>>(
          "children_topological_sort");

  // no tiling : "graph -> dpu subgraph -> super_layer subgraph"
  // tiling : "graph-> dpu subgraph -> tile subgraph -> super_layer subgraph"
  // tile subgraph `mc_code` has no `end` instruction
  // tile subgraph name maybe equals super_layer subgraph name,
  // so can't use subgraph->get_subgraph(name) to find subgraph
  auto all_super_layer_subgraph = std::set<const xir::Subgraph*>();
  for (auto& child : children) {
    if (is_tilling_subgraph(child)) {
      auto super_layer_children = child->get_children();
      all_super_layer_subgraph.insert(super_layer_children.begin(),
                                      super_layer_children.end());
    } else {
      all_super_layer_subgraph.insert(child);
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "loading debug code for " << subgraph_->get_graph()->get_name() << " "
      << children.size() << " subgraphs in total "  //
      << all_super_layer_subgraph.size()
      << " all super layer subgraphs in total "  //
      << child_order.size()
      << " children_topological_sort attr subgraph in total ";

  for (const auto& child_name : child_order) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "load debug code from subgraph[" << child_name << "] " << child_name;

    auto child_subg = std::find_if(
        all_super_layer_subgraph.begin(), all_super_layer_subgraph.end(),
        [&child_name](auto subg) { return subg->get_name() == child_name; });
    UNI_LOG_CHECK(child_subg != all_super_layer_subgraph.end(),
                  VART_XMODEL_ERROR)
        << "cannot find subg " << child_name;
    auto has_mc_code = (*child_subg)->has_attr("mc_code");
    if (has_mc_code) {
      auto& mc_code = (*child_subg)->get_attr<std::vector<char>>("mc_code");
      load_code(DpuReg{"REG_CODE", RegType::CODE, std::move(mc_code)});
      super_layer_subgraph_.emplace_back(*child_subg);
    } else {
      LOG(INFO) << "child_name " << child_name << " has no mc_code";
    }
  }
}
// reg[id] where id is not meaningful
static std::vector<DpuReg> create_workspaces(const xir::Subgraph& subgraph) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "get workspace sizes for " << subgraph.get_graph()->get_name();
  UNI_LOG_CHECK(subgraph.has_attr("reg_id_to_context_type"), VART_XMODEL_ERROR);
  UNI_LOG_CHECK(subgraph.has_attr("reg_id_to_size"), VART_XMODEL_ERROR);
  auto reg_id_to_size =
      subgraph.get_attr<std::map<std::string, int>>("reg_id_to_size");
  auto reg_id_to_context_type =
      subgraph.get_attr<std::map<std::string, std::string>>(
          "reg_id_to_context_type");
  auto total = 0u;
  auto ret = std::vector<DpuReg>{};
  for (auto& reg : reg_id_to_context_type) {
    auto reg_id = reg.first;
    auto reg_type = reg.second;
    if (reg_type != "DATA") {
      continue;
    }
    auto it_size = reg_id_to_size.find(reg_id);
    UNI_LOG_CHECK(it_size != reg_id_to_size.end(), VART_SIZE_MISMATCH);
    auto size = it_size->second;
    ret.emplace_back(reg_id, size);
    total = total + size;
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "total workspace size = " << total;
  UNI_LOG_CHECK(total > 0u, VART_SIZE_MISMATCH)
      << "workspace size must not empty";
  return ret;
}

const xir::Subgraph* DpuKernel::get_subgraph1(size_t idx) const {
  if (super_layer_subgraph_.empty()) {
    // release mode
    UNI_LOG_CHECK(idx == 0u, VART_XMODEL_ERROR) << "LOGICIAL ERROR";
    return subgraph_;
  }
  UNI_LOG_CHECK(idx < super_layer_subgraph_.size(), VART_OUT_OF_RANGE)
      << "LOGICAL ERROR";
  return super_layer_subgraph_[idx];
}

const uint64_t DpuKernel::get_fingerprint() const {
  if (subgraph_->has_attr("dpu_fingerprint")) {
    auto fingerprint = subgraph_->get_attr<std::uint64_t>("dpu_fingerprint");
    return fingerprint;
  }
  return 0u;
}

}  // namespace dpu
}  // namespace vart
