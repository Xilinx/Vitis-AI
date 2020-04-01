/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Kernel Functions Implementation

#include <vector>
#include <iostream>

#include "ext/AksParamValues.h"
#include "ext/AksKernelBase.h"
#include "ext/AksDataDescriptor.h"

using namespace AKS;

class AddKernelBase: public AKS::KernelBase {
  public:
    int exec_async (
        std::vector<AKS::DataDescriptor *> &in,
        std::vector<AKS::DataDescriptor *> &out,
        AKS::OpParamValues* params,
        AKS::DynamicParamValues* dynParams);
    int getNumCUs(void);
};

extern "C" {
AKS::KernelBase* getKernel (AKS::OpParamValues* params) {
  AddKernelBase* base = new AddKernelBase();
  return base;
}
} // extern C

int AddKernelBase::getNumCUs(void)
{
  return 1;
}

int AddKernelBase::exec_async (
  vector<AKS::DataDescriptor *>& in, vector<AKS::DataDescriptor *>& out,
  AKS::OpParamValues* params, AKS::DynamicParamValues* dynParams)
{
    AddKernelBase* kbase = this;
    float* input = (float*)(in[0]->data());

    // Create one output buffer and resize buffer to required size
    out.push_back(new AKS::DataDescriptor({1}, AKS::DataType::FLOAT32));
    float* output = (float*)(out[0]->data());

    output[0] = input[0] + params->_intParams["adder"];
    std::cout << "Node Output : " << output[0] << std::endl;
    return -1;
}
