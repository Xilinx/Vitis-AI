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
#ifndef __LSTM_XRNN__
#define __LSTM_XRNN__

#include <string>
#include <vector>
#include "xxrnn.h"

//#define LSTM_TIME

class xrnn {
 public:
  xrnn(char* model_name);
  ~xrnn();
  //void lstm_create(char* model_name);
  void lstm_run(char* input, int in_size, char* output, int out_size, int frame_num);
  void rnnt_reflash_ddr();
  void rnnt_update_ddr(char * in, int in_s, unsigned int offset);
  void rnnt_download_ddr(char * out, int out_s, unsigned int offset);
  //void lstm_free();
 private:
  xrnn_t xrnn_dev;
  std::vector<char> mbuf;
  std::vector<char> out;
  //char* in;
  //char* out;
  //char* mbuf;
  int rc;
  int out_size;
  std::string model_name_;

};

#endif
