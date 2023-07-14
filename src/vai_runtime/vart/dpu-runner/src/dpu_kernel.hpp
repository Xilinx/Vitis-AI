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
#pragma once
#include <functional>
#include <vector>
#include <vitis/ai/with_injection.hpp>
#include <xir/graph/graph.hpp>

#include "./graph_holder.hpp"
#include "dpu_reg.hpp"
namespace vart {
namespace dpu {
class DpuKernel {
 public:
  // constructor for Vitis AI libraries, where no class holding the graph.
  explicit DpuKernel(const std::string& filename, const std::string& kernel);
  // constructor for VAIE, where the engine hold the graph.
  explicit DpuKernel(const xir::Subgraph& sg, xir::Attrs* attrs);

  virtual ~DpuKernel();

 public:
  // key: "REG_0", "REG_1", or "REG_2" etc
  // TODO: rename
  virtual std::map<std::string, uint64_t> get_parameter(
      size_t device_core_id) const = 0;

  struct SubgraphCode {
    const xir::Subgraph* subgraph;
    uint64_t code_addr;
  };
  virtual std::vector<SubgraphCode> get_code(size_t device_core_id) const = 0;

 protected:
  virtual void load_parameter(const std::vector<DpuReg>& parameters) = 0;
  virtual void load_code(const DpuReg& code) = 0;
  // ret[id] id has no meansing
 public:
  // TODO clean it
  const std::vector<DpuReg>& get_workspace_regs() const {
    return workspace_regs_;
  }
  // TODO DELETE IT
  size_t get_num_of_codes() const { return super_layer_subgraph_.size(); }
  // TODO DELETE IT
  const xir::Subgraph* get_subgraph1(size_t idx) const;
  // ok
  const xir::Subgraph* get_subgraph() const { return subgraph_; }

  const uint64_t get_fingerprint() const;

 private:
  void my_load_parameter();
  void my_load_release_code();
  void my_load_debug_code();
  void init_reg_info();

 public:
  virtual void initialize();

 private:
  // graph with same filename will share the same graph holder
  // just for Vitis AI library
  std::shared_ptr<GraphHolder> graph_holder_;

 private:
  const xir::Subgraph* subgraph_;
  // workspace_sizes_[id],  id has no meaning.
  const std::vector<DpuReg> workspace_regs_;

 protected:
  std::vector<const xir::Subgraph*> super_layer_subgraph_;
};
}  // namespace dpu
}  // namespace vart
