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



#include <iostream>

#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
#include "xir/graph/graph.hpp"
// xrt.h must be included after. otherwise, name pollution.
#include <xrt.h>
using namespace std;
int main(int argc, char* argv[]) {
  LOG(INFO) << "HELLO , testing is started";
  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "DPU") {
      s = c;
      break;
    }
  }
  auto h = xclOpen(0, NULL, XCL_INFO);
  auto input_tensor_buffer_size = vart::get_input_buffer_size(s);
  auto bo1 = xclAllocBO(h, input_tensor_buffer_size, 0, 0);
  auto tensors = s->get_sorted_output_tensors();
  CHECK(!tensors.empty());
  auto tensor = *tensors.begin();
  tensor = tensor->get_producer()->get_input_op("input")->get_output_tensor();
  LOG(INFO) << "tensor = " << tensor->to_string();
  auto tensor_buffer = vart::assistant::XrtBoTensorBuffer::create(
      vart::xrt_bo_t{h, bo1}, tensor);
  LOG(INFO) << "tensor_buffer=" << tensor_buffer->to_string();
  auto data_phy = tensor_buffer->data_phy({0, 0, 0, 0});
  LOG(INFO) << "phy = " << std::hex << "0x" << data_phy.first << std::dec
            << " size=" << data_phy.second
            << " bo_size= " << input_tensor_buffer_size;
  // clean up
  xclFreeBO(h, bo1);
  xclClose(h);
  CHECK(s != nullptr);
  return 0;
}
