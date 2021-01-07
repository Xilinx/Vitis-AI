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

#pragma once
#include <string>
#include <vector>
#include <memory>

class dpu4rnn {
 public:
  dpu4rnn();
  dpu4rnn(const std::string& model_name) = delete;
  virtual ~dpu4rnn();

  static std::unique_ptr<dpu4rnn> create(const std::string& model_name,
		  const int device_id = 0);
  virtual void run(const char* input, int in_size,
		  char* output, int frame_num, int batch = 1) = 0;
  virtual int getBatch() = 0;
};
