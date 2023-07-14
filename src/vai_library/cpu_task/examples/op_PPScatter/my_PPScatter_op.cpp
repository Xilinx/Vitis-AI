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
#include <vart/op_imp.h>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(ENABLE_BENCHMARK_RUN, "1")
class MyPPScatterOp {
 public:
  MyPPScatterOp(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {
    b_benchmarkrun = ENV_PARAM(ENABLE_BENCHMARK_RUN);
    // op and attrs is not in use.
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                 std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
     CHECK_EQ(inputs.size(), 2);
     auto input_data_shape = inputs[0].tensor->get_shape();
     auto input_coord_shape = inputs[1].tensor->get_shape();
     auto output_shape = output.tensor->get_shape();
     CHECK_EQ(input_data_shape.size(), 4); // 1 12000  1 64  --> 1 64 12000 1
     CHECK_EQ(input_coord_shape.size(), 3); // 1  12000 4
     CHECK_EQ(output_shape.size(), 4);   // 1 496 432 64  ---> 1 64 496 432
 
     auto coord_numbers = input_coord_shape[1];
     auto coord_channel = input_coord_shape[2];
     CHECK_EQ(coord_numbers, input_data_shape[2]);
 
     auto batch = output_shape[0];
     auto height = output_shape[2];
     auto width = output_shape[3];
     auto channel = output_shape[1];
     CHECK_EQ(input_data_shape[0], batch);
     CHECK_EQ(channel, input_data_shape[1]);
 
     auto output_idx = 0;
     auto input_idx = 0;
     auto x_idx = 0;
 
     memset(output.data, 0, output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3]*sizeof(float));
 
     for (auto n = 0; n < coord_numbers; n++) {
       auto x = (int)inputs[1].data[x_idx + 3];
       auto y = (int)inputs[1].data[x_idx + 2];
       if (b_benchmarkrun) {
          if (x >= width || y >= height) {
            LOG(WARNING) <<"x, y exceed limit, maybe in benchmark mode?";
            return 0;
          }
       }
       if (x < 0) break;  // stop copy data when coord x == -1 .
       for(int i=0; i<channel; i++) {
          output_idx =i*height*width + y*width+x;
          input_idx = n+i*coord_numbers;
          output.data[output_idx] = inputs[0].data[ input_idx ];
       }
       x_idx += coord_channel;
     }
     return 0;
  }

 public:
  const xir::Op* const op;
  bool b_benchmarkrun;
};

DEF_XIR_OP_IMP(MyPPScatterOp)
