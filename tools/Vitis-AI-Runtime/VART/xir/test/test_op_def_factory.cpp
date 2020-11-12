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

#include "xir/attrs/attrs.hpp"
#include "xir/op/op_def_factory_imp.hpp"

#include "UniLog/UniLog.hpp"

using namespace std;
using namespace xir;

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);

  auto op_list = op_def_factory()->get_registered_ops();
  for (auto op : op_list) {
    cout << op << endl;
  }

  auto param = Attrs::create();
  param->set_attr<int>("kernel_h", 3);
  cout << "kernel_h " << param->get_attr<int>("kernel_h") << endl;

  return 0;
}
