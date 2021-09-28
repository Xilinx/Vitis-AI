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
/**
 *  @brief type definitions used in xf_blas library
 *
 *  $DateTime: 2019/06/13 $
 */
#ifndef BLAS_DEF_HPP
#define BLAS_DEF_HPP
#include "blas_gen_bin.hpp"

/*#define BLAS_pageSizeBytes 4096
#define BLAS_instrSizeBytes 8
#define BLAS_instrPageIdx 0
#define BLAS_paramPageIdx 1
#define BLAS_stasPageIdx 2
#define BLAS_dataPageIdx 3
#define BLAS_maxNumInstr 64
#define BLAS_dataType int*/
namespace xf {

namespace blas {

typedef GenBin<BLAS_dataType,
               BLAS_resDataType,
               void*,
               BLAS_memWidthBytes,
               BLAS_parEntries,
               BLAS_instrSizeBytes,
               BLAS_pageSizeBytes,
               BLAS_maxNumInstrs,
               BLAS_instrPageIdx,
               BLAS_paramPageIdx,
               BLAS_statsPageIdx>
    GenBinType;

} // end namespace blas

} // end namespace xf

#endif
