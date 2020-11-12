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

#include "UniLog/UniLog.hpp"
#include "iostream"
#include "xir/tensor/tensor.hpp"
#include "xir/tensor/tensor_imp.hpp"

int main() {
  auto t =
      xir::Tensor::create("helloworld", {1, 2, 3, 4}, xir::DataType{"INT8"});
  auto data_type = t->get_data_type();
  UNI_LOG_CHECK(t->get_name() == "helloworld", XIR_MEANINGLESS_VALUE);
  UNI_LOG_CHECK(t->get_shape().size() == 4, XIR_MEANINGLESS_VALUE);
  UNI_LOG_CHECK(t->get_shape().at(0) == 1, XIR_MEANINGLESS_VALUE);
  UNI_LOG_CHECK(t->get_element_num() == 24, XIR_MEANINGLESS_VALUE);
  UNI_LOG_CHECK(data_type.bit_width == 8, XIR_MEANINGLESS_VALUE);

  auto same0 =
      xir::Tensor::create("helloworld", {1, 2, 3, 4}, xir::DataType{"FLOAT32"});
  auto same1 =
      xir::Tensor::create("helloworld", {1, 2, 3, 4}, xir::DataType{"FLOAT64"});
  auto diff0 =
      xir::Tensor::create("helloworld", {2, 3, 4, 5}, xir::DataType{"XINT32"});
  auto diff1 = xir::Tensor::create("helloworld", {-1, 2, 3, 4},
                                   xir::DataType{"FLOAT64"});

  UNI_LOG_CHECK(t->get_data_type().type == xir::DataType::INT,
                XIR_MEANINGLESS_VALUE);

  auto strange = xir::Tensor::create("helloworld", {5}, xir::DataType{"INT4"});
  UNI_LOG_CHECK(strange->get_data_size() == 3, XIR_MEANINGLESS_VALUE);
}

// int main() { return 0; }
