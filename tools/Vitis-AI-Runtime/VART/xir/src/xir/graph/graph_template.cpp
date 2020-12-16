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

#include "xir/graph/graph_template.hpp"
#include "xir/graph/graph_template_imp.hpp"

namespace xir {

std::unique_ptr<GraphTemplate> GraphTemplate::create(std::string name) {
  return std::unique_ptr<GraphTemplate>{
      static_cast<GraphTemplate*>(new GraphTemplateImp{name})};
}

}  // namespace xir
