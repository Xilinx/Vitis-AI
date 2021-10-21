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
 * @file selMultXkernel.cpp
 * @brief selMultXkernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "selMultXkernel.hpp"
extern "C" void selMultXkernel(
    ParamStrTyp& p_paramStr, HBM_StrTyp& p_xStr, HBM_StrTyp& p_nnzStr, DatStrTyp& p_outDatStr, IdxStrTyp& p_idxStr) {
    AXIS(p_paramStr)
    AXIS(p_xStr)
    AXIS(p_nnzStr)
    AXIS(p_outDatStr)
    AXIS(p_idxStr)
    AP_CTRL_NONE(return )
#pragma HLS DATAFLOW

    xf::sparse::selMultAddX<SPARSE_dataType, SPARSE_indexType, SPARSE_maxCols, SPARSE_parEntries, SPARSE_accLatency,
                            SPARSE_dataBits, SPARSE_indexBits>(p_paramStr, p_xStr, p_nnzStr, p_outDatStr, p_idxStr);
}
