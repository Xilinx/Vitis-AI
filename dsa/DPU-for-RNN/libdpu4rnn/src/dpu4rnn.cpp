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
#include "dpu4rnn.hpp"
#include "dpu4rnn_imp.hpp"

dpu4rnn::dpu4rnn() {}
dpu4rnn::~dpu4rnn() {}

std::unique_ptr<dpu4rnn> dpu4rnn::create(const std::string& model_name, const int device_id) {
  //std::cout << "model name is " << model_name << std::endl;
  return std::unique_ptr<dpu4rnn>(new dpu4rnnImp(model_name, device_id));
}

