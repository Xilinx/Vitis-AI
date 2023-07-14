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
#include "vitis/ai/env_config.hpp"
#include <vart/op_imp.h>
DEF_ENV_PARAM(HACK_TRANSPOSE, "0")
class MyMatmulOp {
 public:
  MyMatmulOp(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {
    transpose_a = op->get_attr<bool>("transpose_a");
    transpose_b = op->get_attr<bool>("transpose_b");
    transpose_b = transpose_b || ENV_PARAM(HACK_TRANSPOSE);
  }

  int calculate(vart::simple_tensor_buffer_t<float> output,
                 std::vector<vart::simple_tensor_buffer_t<float>> inputs) {

     CHECK_EQ(inputs.size(), 2);
     auto input_shape_a = inputs[0].tensor->get_shape();
     auto input_shape_b = inputs[1].tensor->get_shape();
     auto output_shape = output.tensor->get_shape();
     CHECK(input_shape_a.size()>=2);
     CHECK(input_shape_b.size()>=2);
     CHECK(output_shape.size()>=2);
     CHECK(output_shape.size() == input_shape_a.size());

     // std::cout <<"inshapea:" <<input_shape_a.size() << "  - " << input_shape_a[0] << " " << input_shape_a[1] << " " << input_shape_a[2] << " " << input_shape_a[3] <<"\n"; // inshapea:1 1 36608 3
     M_ = input_shape_a[input_shape_a.size() - (transpose_a?1:2)];
     K_ = input_shape_a[input_shape_a.size() - (transpose_a?2:1)];
     N_ = input_shape_b[input_shape_b.size() - (transpose_b?2:1)];
     CHECK(K_ == input_shape_b[input_shape_b.size() - (transpose_b?1:2)]);

     batch_a_ = get_batch(input_shape_a, 2);
     batch_b_ = get_batch(input_shape_b, 2);

     int total = get_batch(output_shape, 0);
     int idx_a = 0, idx_b = 0;
     memset(output.data, 0, total*sizeof(float));
     int B_ = get_batch(output_shape, 2);

     for (int b=0; b<B_; b++) {
       for (int m=0; m<M_; m++) {
         for (int n=0; n<N_; n++) {
           for(auto k=0; k<K_; k++) {
             // D[m*N+n] += Sa[m*K+k] * Sb[k*N+n];
             idx_a = (b%batch_a_)*M_*K_ + (transpose_a ? k*M_+m : m*K_+k);
             idx_b = (b%batch_b_)*K_*N_ + (transpose_b ? n*K_+k : k*N_+n);
             output.data[b*M_*N_ + m*N_+n] += inputs[0].data[idx_a] * inputs[1].data[idx_b];
           }
         }
       }
     }
     // std::cout <<"outshape:" << output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << " " << output_shape[3] << "\n";  // 1 1 36608 3
     return 0;
  }

 private:
  int get_batch(std::vector<int>& v, int cut) {
     int ret = 1;
     for(int i=0; i<(int)v.size()-cut; i++) {
       ret*=v[i];
     }
     return ret;
  }
 public:
  const xir::Op* const op=nullptr;
  bool transpose_a=false;
  bool transpose_b=false;
  // 1st matrix is MxK
  // 2nd matrix is KxN
  // result is MxN
  // M, K, N are all transposed value.
  int M_=0;
  int K_=0;
  int N_=0;
  // because  input a and input b support broadcast,
  // so batch_a_ and batch_b_ are different.
  int batch_a_=0;
  int batch_b_=0;

};

DEF_XIR_OP_IMP(MyMatmulOp)

