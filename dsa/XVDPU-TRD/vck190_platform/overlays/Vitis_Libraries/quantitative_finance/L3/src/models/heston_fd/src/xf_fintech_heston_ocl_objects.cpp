/*
 * Copyright 2019 Xilinx, Inc.
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

#include "xf_fintech_heston_ocl_objects.hpp"

using namespace xf::fintech;

HestonFDOCLObjects::HestonFDOCLObjects() {}

HestonFDOCLObjects::HestonFDOCLObjects(cl::Context* pContext, cl::CommandQueue* pCommandQueue, cl::Kernel* pKernel) {
    m_pContext = pContext;
    m_pCommandQueue = pCommandQueue;
    m_pKernel = pKernel;
}

cl::Context* HestonFDOCLObjects::GetContext() {
    return m_pContext;
}

cl::CommandQueue* HestonFDOCLObjects::GetCommandQueue() {
    return m_pCommandQueue;
}

cl::Kernel* HestonFDOCLObjects::GetKernel() {
    return m_pKernel;
}
