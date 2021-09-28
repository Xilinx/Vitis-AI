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
 * @file krnl_loadPkApar.cpp
 * @brief krnl_loadPkApar definition.
 *
 * This file is part of Vitis HPC Library.
 */

#include "krnl_loadPkApar.hpp"

extern "C" void krnl_loadPkApar(CG_interface* p_parParamPtr,
                                CG_interface* p_xPtr,
                                CG_tkStrType& p_tkInStr,
                                CG_wideParamStrType& p_paramStr,
                                CG_wideStrType& p_outXstr) {
    POINTER(p_parParamPtr, p_parParamPtr)
    POINTER(p_xPtr, p_xPtr)
    AXIS(p_tkInStr)
    AXIS(p_paramStr)
    AXIS(p_outXstr)
    SCALAR(return )

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tkInStr, l_cs);
    while (!l_token.getExit()) {
        xf::sparse::loadParX<SPARSE_indexType, CG_numChannels, SPARSE_indexBits, SPARSE_hbmMemBits>(
            p_parParamPtr, p_xPtr, p_paramStr, p_outXstr);
        l_token.read_decode(p_tkInStr, l_cs);
    }
}
