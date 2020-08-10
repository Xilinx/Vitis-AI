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

#ifndef _DPU_RESIZE_H_
#define _DPU_RESIZE_H_

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <vector>

using namespace std;

struct _param {
  uint16_t start_x;
  uint16_t end_x;
  uint16_t start_y;
  uint16_t end_y;
  uint16_t frac_x[2];
  uint16_t frac_y[2];
};
struct _config {
  uint32_t scale_w;
  uint32_t scale_h;
  uint16_t src_w;
  uint16_t src_h;
  uint16_t src_c;
  uint16_t dst_w;
  uint16_t dst_h;
  uint16_t dst_c;
  uint16_t inter_mode;
};

class dpu_resize {
private:
  int CQBIT;
  int IM_LINEAR;
  int IM_NEAREST;
  int IM_MAX;

  struct _config cfg;
  struct _param **p_matrix;
  uint8_t *img_src;
  uint8_t *img_dst;

  void param_gen();

public:
  dpu_resize(uint8_t *ext_img_src, uint8_t *exi_img_dst,
             struct _config ext_cfg);
  void calc();
  void dump_config();
  void dump_img_src();
  void dump_img_dst();
  void dump_param();
};

#endif
