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
 * @file loadParXkernel.cpp
 * @brief loadParXkernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "loadParXkernel.hpp"

extern "C" void loadParXkernel(HBM_InfTyp* p_parParamPtr,
                               HBM_InfTyp* p_xPtr,
                               WideParamStrTyp& p_paramStr,
                               HBM_StrTyp& p_outXstr) {
    POINTER(p_parParamPtr, p_parParamPtr)
    POINTER(p_xPtr, p_xPtr)
    AXIS(p_paramStr)
    AXIS(p_outXstr)
    SCALAR(return )

    xf::sparse::loadParX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_indexBits, SPARSE_hbmMemBits>(
        p_parParamPtr, p_xPtr, p_paramStr, p_outXstr);
}
