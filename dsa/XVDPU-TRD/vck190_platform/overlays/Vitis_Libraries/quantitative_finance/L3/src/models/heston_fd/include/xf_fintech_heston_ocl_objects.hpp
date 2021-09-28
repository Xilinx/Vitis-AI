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
#ifndef _XF_FINTECH_HESTON_OCL_OBJECTS_H_
#define _XF_FINTECH_HESTON_OCL_OBJECTS_H_

#include "xcl2.hpp"

namespace xf {
namespace fintech {

class HestonFDOCLObjects {
   public:
    HestonFDOCLObjects();

    HestonFDOCLObjects(cl::Context* pContext, cl::CommandQueue* pCommandQueue, cl::Kernel* pKernel);

    cl::Context* GetContext();
    cl::CommandQueue* GetCommandQueue();
    cl::Kernel* GetKernel();

   private:
    cl::Context* m_pContext;
    cl::CommandQueue* m_pCommandQueue;
    cl::Kernel* m_pKernel;
};
} // namespace fintech
} // namespace xf

#endif /* _XF_FINTECH_HESTON_OCL_OBJECTS_H_ */
