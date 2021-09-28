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
 * @file krnl_loadArbParam.cpp
 * @brief krnl_loadArbParam definition.
 *
 * This file is part of Vitis HPC Library.
 */

#include "krnl_loadArbParam.hpp"

extern "C" void krnl_loadArbParam(CG_interface* p_rbParamPtr,
                                  CG_tkStrType& p_tkInStr,
                                  CG_wideParamStrType& p_chRbParamStr,
                                  CG_paramStrType& p_rbParamStr) {
    POINTER(p_rbParamPtr, p_rbParamPtr)
    AXIS(p_tkInStr)
    AXIS(p_chRbParamStr)
    AXIS(p_rbParamStr)
    SCALAR(return )

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tkInStr, l_cs);
    while (!l_token.getExit()) {
        xf::sparse::loadRowParam<CG_numChannels, SPARSE_hbmMemBits>(p_rbParamPtr, p_rbParamStr, p_chRbParamStr);
        l_token.read_decode(p_tkInStr, l_cs);
    }
}
