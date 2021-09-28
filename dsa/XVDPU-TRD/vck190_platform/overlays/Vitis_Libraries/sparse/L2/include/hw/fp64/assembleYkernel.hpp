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
 * @file assembleYkernel.hpp
 * @brief assembleYkernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_ASSEMBLEYKERNEL_HPP
#define XF_SPARSE_ASSEMBLEYKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief assembleYkernel is used to assemble the accumulated results into Y
 * @param p_paramStr input axis stream of row block parameters
 * @param p_datStr input axis stream of accumulated results
 * @param p_yStr output axis stream of Y
 */
extern "C" void assembleYkernel(ParamStrTyp& p_paramStr, DatStrTyp& p_datStr, DatStrTyp& p_yStr);

#endif
