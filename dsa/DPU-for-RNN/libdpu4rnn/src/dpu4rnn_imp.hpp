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
#include <mutex>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

#include "../include/dpu4rnn.hpp"

class dpu4rnnImp : public dpu4rnn {
 public:
  dpu4rnnImp(const std::string model_name, const int device_id = 0);
  virtual ~dpu4rnnImp();

  virtual void run(const char* input, int in_size, char* output,
		  int frame_num, int batch = 1) override;
  virtual int getBatch() override;

 private:
  void set(int batch, int frames, int in_len, int out_len) {
    idims_ = {batch, frames, in_len, 2};
    odims_ = {batch, frames, out_len, 2};
  };

  std::vector<int32_t> get_input_dims() {return idims_;};
  std::vector<int32_t> get_output_dims() {return odims_;};

 private:
  std::shared_ptr<xir::Graph> fakegraph;
  xir::Subgraph* rs;
  std::unique_ptr<vart::Runner> runner_;

  int device_id_;
  int batch_;
  std::string model_name_;

  std::vector<int32_t> idims_;
  std::vector<int32_t> odims_;
  std::vector<char> model_;
  
  std::vector<char> out_;

  std::mutex mtx;

  static const std::string xclbin_path_;
  static const std::vector<std::string> model_type_;
};

