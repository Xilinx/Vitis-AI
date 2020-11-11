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
#include <iostream>
#include <vector>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

class BridgeKernel : public AKS::KernelBase
{
  public:
    int exec_async (
           std::vector<AKS::DataDescriptor*> &in, 
           std::vector<AKS::DataDescriptor*> &out, 
           AKS::NodeParams* nodeParams,
           AKS::DynamicParamValues* dynParams);
};

extern "C" { // Add this to make this available for python bindings

AKS::KernelBase* getKernel (AKS::NodeParams *params)
{
    return new BridgeKernel();
}

} //extern "C"

int BridgeKernel::exec_async (
                      std::vector<AKS::DataDescriptor*> &in, 
                      std::vector<AKS::DataDescriptor*> &out, 
                      AKS::NodeParams* nodeParams,
                      AKS::DynamicParamValues* dynParams)
{
    for (auto& input: in) {
        out.push_back(new AKS::DataDescriptor(*input));
    }
    // std::cout << "[DBG] BridgeKernel: Done!" << std::endl << std::endl;
    return -1; // No wait
}

