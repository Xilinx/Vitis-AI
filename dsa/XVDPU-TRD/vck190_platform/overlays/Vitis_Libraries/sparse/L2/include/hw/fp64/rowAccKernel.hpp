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
 * @file rowAccKernel.hpp
 * @brief accumulate intermediate results along rows.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_ROWACCKERNEL_HPP
#define XF_SPARSE_ROWACCKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief rowAccKernel is accumulate the data along row indices
 * @param p_paramStr input axis stream of row block parameters
 * @param p_inDatStr input axis stream of multiplication and partially accumulated results
 * @param p_idxStr input axis stream of row indices
 * @param p_outDatStr output axis stream of accumulation results
 */
extern "C" void rowAccKernel(WideParamStrTyp& p_paramStr,
                             DatStrTyp& p_inDatStr0,
                             DatStrTyp& p_inDatStr1,
                             DatStrTyp& p_inDatStr2,
                             DatStrTyp& p_inDatStr3,
                             DatStrTyp& p_inDatStr4,
                             DatStrTyp& p_inDatStr5,
                             DatStrTyp& p_inDatStr6,
                             DatStrTyp& p_inDatStr7,
                             DatStrTyp& p_inDatStr8,
                             DatStrTyp& p_inDatStr9,
                             DatStrTyp& p_inDatStr10,
                             DatStrTyp& p_inDatStr11,
                             DatStrTyp& p_inDatStr12,
                             DatStrTyp& p_inDatStr13,
                             DatStrTyp& p_inDatStr14,
                             DatStrTyp& p_inDatStr15,

                             IdxStrTyp& p_idxStr0,
                             IdxStrTyp& p_idxStr1,
                             IdxStrTyp& p_idxStr2,
                             IdxStrTyp& p_idxStr3,
                             IdxStrTyp& p_idxStr4,
                             IdxStrTyp& p_idxStr5,
                             IdxStrTyp& p_idxStr6,
                             IdxStrTyp& p_idxStr7,
                             IdxStrTyp& p_idxStr8,
                             IdxStrTyp& p_idxStr9,
                             IdxStrTyp& p_idxStr10,
                             IdxStrTyp& p_idxStr11,
                             IdxStrTyp& p_idxStr12,
                             IdxStrTyp& p_idxStr13,
                             IdxStrTyp& p_idxStr14,
                             IdxStrTyp& p_idxStr15,

                             DatStrTyp& p_outDatStr);

#endif
