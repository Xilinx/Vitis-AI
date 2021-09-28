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
/**
 * @file loadRbParamKernel.cpp
 * @brief loadRbParamKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "loadRbParamKernel.hpp"

extern "C" void loadRbParamKernel(HBM_InfTyp* p_rbParamPtr,
                                  WideParamStrTyp& p_chRbParamStr,
                                  ParamStrTyp& p_rbParamStr) {
    POINTER(p_rbParamPtr, p_rbParamPtr)
    AXIS(p_chRbParamStr)
    AXIS(p_rbParamStr)
    SCALAR(return )

    xf::sparse::loadRowParam<SPARSE_hbmChannels, SPARSE_hbmMemBits>(p_rbParamPtr, p_rbParamStr, p_chRbParamStr);
}
