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
 * @file selMultXkernel.hpp
 * @brief select and multiply input vector X.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_SELMULTXKERNEL_HPP
#define XF_SPARSE_SELMULTXKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief selMultXkernel is used to select and multiply input vector X with NNZ values
 * @param p_paramStr input axis stream of partition parameters
 * @param p_xStr input axis stream of X entries
 * @param p_nnzStr input axis stream of NNZ values and indices
 * @param p_outDatStr output axis stream of multiplication results
 * @param p_idxStr output row indices stream of the partially accumulated results
 */
extern "C" void selMultXkernel(
    ParamStrTyp& p_paramStr, HBM_StrTyp& p_xStr, HBM_StrTyp& p_nnzStr, DatStrTyp& p_outDatStr, IdxStrTyp& p_idxStr);

#endif
