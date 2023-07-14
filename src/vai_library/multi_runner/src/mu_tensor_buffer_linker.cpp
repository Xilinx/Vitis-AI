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
#include "mu_tensor_buffer_linker.hpp"

#include <memory>
#include <sstream>

std::unique_ptr<MUTensorBufferLinker> MUTensorBufferLinker::create(
    std::unique_ptr<vart::TensorBuffer>* master) {
  return std::make_unique<MUTensorBufferLinker>(master);
}

MUTensorBufferLinker::MUTensorBufferLinker(
    std::unique_ptr<vart::TensorBuffer>* master)
    : master_{master} {}

MUTensorBufferLinker::~MUTensorBufferLinker() {}

std::string MUTensorBufferLinker::to_string() {
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

// void MUTensorBufferLinker::finalize() {
//  LOG(ERROR) << " please override this function";
//}

void MUTensorBufferLinker::after_invoke_runner(const xir::Subgraph* subgraph) {
  uint64_t data_from = 0u;
  size_t data_size = 0;
  std::tie(data_from, data_size) = master_->get()->data(
      std::vector<int>(master_->get()->get_tensor()->get_shape().size(), 0));
  for (auto s : slaves_) {
    uint64_t data_to = 0u;
    size_t data_size_to = 0;
    std::tie(data_to, data_size_to) = s->get()->data(
        std::vector<int>(s->get()->get_tensor()->get_shape().size(), 0));
    CHECK_GE(data_size_to, data_size);
    memcpy((void*)data_to, (void*)data_from, data_size);
  }
}
