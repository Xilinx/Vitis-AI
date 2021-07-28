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
#include "./tensor_buffer_linker.hpp"

#include <memory>
#include <sstream>

#include "./tensor_buffer_linker_p.hpp"
#include "./tensor_buffer_linker_v.hpp"

std::unique_ptr<TensorBufferLinker> TensorBufferLinker::create(
    std::unique_ptr<vart::TensorBuffer>* master) {
  auto ret = std::unique_ptr<TensorBufferLinker>();
  switch (master->get()->get_location()) {
    case vart::TensorBuffer::location_t::HOST_VIRT:
      ret = std::make_unique<TensorBufferLinkerHostVirt>(master);
      break;
    case vart::TensorBuffer::location_t::HOST_PHY:
      ret = std::make_unique<TensorBufferLinkerHostPhy>(master);
      break;
    default:
      LOG(FATAL) << "Not supported yet";
  }
  CHECK(ret != nullptr) << "Not supported yet";
  return ret;
}

TensorBufferLinker::TensorBufferLinker(
    std::unique_ptr<vart::TensorBuffer>* master)
    : master_{master} {}

TensorBufferLinker::~TensorBufferLinker() {}

std::string TensorBufferLinker::to_string() {
  std::ostringstream str;
  str << "linker{master=" << master_->get()->to_string() << "; slaves=[";
  int c = 0;
  for (auto s : slaves_) {
    if (c != 0) {
      str << ",";
    }
    str << s->get()->to_string();
    c++;
  }
  str << "]}";
  return str.str();
}

void TensorBufferLinker::finalize(std::string device) {
  // LOG(ERROR) << " please override this function";
}

void TensorBufferLinker::after_invoke_runner(const xir::Subgraph* subgraph) {
  LOG(WARNING) << " not so efficient, too much copying";
  for (auto s : slaves_) {
    vart::TensorBuffer::copy_tensor_buffer(master_->get(), s->get());
  }
}
