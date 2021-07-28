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

#include <aks/AksTensorBuffer.h>
#include <aks/AksNodeParams.h>
#include <aks/AksKernelBase.h>

class AddKernelBase: public AKS::KernelBase {
  public:
    int exec_async (
        std::vector<vart::TensorBuffer *> &in,
        std::vector<vart::TensorBuffer *> &out,
        AKS::NodeParams* params,
        AKS::DynamicParamValues* dynParams);
    int getNumCUs(void);
};

extern "C" {
  AKS::KernelBase* getKernel (AKS::NodeParams* params) {
    AddKernelBase* base = new AddKernelBase();
    return base;
  }
} // extern C

int AddKernelBase::getNumCUs(void)
{
  return 1;
}

int AddKernelBase::exec_async (
    vector<vart::TensorBuffer *>& in, vector<vart::TensorBuffer *>& out,
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams)
{
  AddKernelBase* kbase = this;
  float* input = reinterpret_cast<float*>(in[0]->data().first);

  // Create one output buffer and resize buffer to required size
  std::string tensorName ("add-out");
  out.push_back(new AKS::AksTensorBuffer(xir::Tensor::clone(in[0]->get_tensor())));
  float* output = reinterpret_cast<float*>(out[0]->data().first);
  // Add 
  output[0] = input[0] + params->_intParams["adder"];

  return 0;
}
