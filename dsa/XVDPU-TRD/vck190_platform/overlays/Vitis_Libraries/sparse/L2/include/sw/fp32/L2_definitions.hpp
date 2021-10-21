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
 * @file L2_definitions.hpp
 * @brief common definitions used in host coce.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_L2_DEFINITIONS_HPP
#define XF_SPARSE_L2_DEFINITIONS_HPP

#include "gen_cscmv.hpp"
#include "mtxFile.hpp"
#include "interpreters.hpp"

namespace xf {
namespace sparse {

static const unsigned int SPARSE_dataSz = sizeof(SPARSE_dataType);
static const unsigned int SPARSE_idxSz = sizeof(SPARSE_indexType);
static const unsigned int SPARSE_NnzRowIdxMemWords = SPARSE_hbmMemBits / (8 * SPARSE_dataSz);
static const unsigned int SPARSE_ColVecMemWords = SPARSE_ddrMemBits / (8 * SPARSE_dataSz);
static const unsigned int SPARSE_NnzPtrMemWords = SPARSE_ddrMemBits / (8 * SPARSE_idxSz);
static const unsigned int SPARSE_ddrPtrSz = SPARSE_ddrMemBits * SPARSE_dataSz / SPARSE_dataBits;
static const unsigned int SPARSE_hbmPtrSz = SPARSE_hbmMemBits * SPARSE_dataSz / SPARSE_dataBits;
static const unsigned int SPARSE_paramPtrSz = 4;
static const unsigned int SPARSE_maxCols = SPARSE_maxColParBlocks * SPARSE_parEntries;
static const unsigned int SPARSE_maxRows = SPARSE_maxRowBlocks * SPARSE_parEntries * SPARSE_parGroups;

// common types
typedef NnzUnit<SPARSE_dataType, SPARSE_indexType> NnzUnitType;
typedef MtxFile<SPARSE_dataType, SPARSE_indexType> MtxFileType;
typedef Program<SPARSE_pageSize> ProgramType;
typedef RunConfig<SPARSE_dataType,
                  SPARSE_indexType,
                  SPARSE_parEntries,
                  SPARSE_parGroups,
                  SPARSE_ddrMemBits,
                  SPARSE_hbmMemBits,
                  SPARSE_hbmChannels,
                  SPARSE_paramOffset,
                  SPARSE_pageSize>
    RunConfigType;

typedef ColVec<SPARSE_dataType, SPARSE_ddrMemBits> ColVecType;

typedef MatCsc<SPARSE_dataType, SPARSE_indexType, SPARSE_pageSize> MatCscType;

typedef GenMatCsc<SPARSE_dataType, SPARSE_indexType> GenMatCscType;

typedef GenVec<SPARSE_dataType, SPARSE_ddrMemBits, SPARSE_pageSize> GenVecType;
typedef CscRowInt<SPARSE_dataType, SPARSE_indexType, SPARSE_parEntries, SPARSE_parGroups> CscRowIntType;
} // end namespace sparse
} // end namespace xf
#endif
