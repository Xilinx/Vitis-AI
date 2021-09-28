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
 * @file storeYkernel.hpp
 * @brief storeYkernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_STOREYKERNEL_HPP
#define XF_SPARSE_STOREYKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief storeYkernel is used to write result Y vector into device memory
 * @param p_rows number of entries in the result Y vector
 * @param p_yStr input axis stream of Y vector entries
 * @param p_yPtr  device memory pointer for writing Y vector
 */
extern "C" void storeYkernel(unsigned int p_rows, HBM_InfTyp* p_yPtr, DatStrTyp& p_yStr);
#endif
