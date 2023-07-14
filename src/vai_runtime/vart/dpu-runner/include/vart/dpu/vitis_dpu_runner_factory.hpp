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
#include <memory>
#include <string>

#include "vart/runner.hpp"
namespace xir {
class Subgraph;
class Attrs;
}  // namespace xir
namespace vart {
namespace dpu {
class DpuRunnerFactory {
 public:
  // this factory method is legacy, and used by TRD and for testing etc.
  static std::unique_ptr<vart::Runner> create_dpu_runner(
      const std::string& file_name, const std::string& kernel_name);
  /// this factory method is used by VAID, the meta info is stored in
  /// the xmodel.
  static std::unique_ptr<vart::Runner> create_dpu_runner(
      const xir::Subgraph* subgraph, xir::Attrs* attrs);
};
}  // namespace dpu
}  // namespace vart
