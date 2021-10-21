/**********
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
 * **********/

#ifndef XF_BLAS_DDRTYPE_HPP
#define XF_BLAS_DDRTYPE_HPP

#include "ddr.hpp"
#include "kargs.hpp"

// Location of code aand data segements in DDR memory
#define BLAS_codePage 0
#define BLAS_resPage 1
#define BLAS_dataPage 2

// Page and instruction sizes
#define BLAS_pageSizeBytes 4096

// DDR interface types
typedef xf::blas::DdrUtil<BLAS_dataType, BLAS_ddrWidth, BLAS_ddrWidth * sizeof(BLAS_dataType) * 8> DdrUtilType;
typedef DdrUtilType::DdrWideType DdrType;
typedef typename DdrType::t_TypeInt DdrIntType;

// VLIV processing types
typedef xf::blas::Kargs<BLAS_dataType, BLAS_ddrWidth, BLAS_argInstrWidth, BLAS_argPipeline> KargsType;

typedef KargsType::DdrInstrType KargsDdrInstrType; // 512 bit wide type across all DDR-width architectures

#endif
