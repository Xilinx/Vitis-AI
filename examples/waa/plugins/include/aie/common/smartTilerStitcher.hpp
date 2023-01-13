/******************************************************************************
 *  Copyright (c) 2018, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

#ifndef _SMARTTILERSTITCHER_HPP_
#define _SMARTTILERSTITCHER_HPP_

#include <common/smartTile.hpp>

namespace xF {
// debug parameters
const bool enableVerbose = false;

void smartTileTilerGenerateMetaDataWithSpecifiedTileSize(std::vector<uint16_t> srcSize,
                                                         std::vector<xF::smartTileMetaData>& metaDataList,
                                                         uint16_t& numberOfTileRows,
                                                         uint16_t& numberOfTileColumns,
                                                         std::vector<uint16_t> maxTileSize,
                                                         std::vector<uint16_t> overlapSizeH = {0, 0},
                                                         std::vector<uint16_t> overlapSizeV = {0, 0},
                                                         int tileAlignment = 1,
                                                         bool enableInvalidRegions = true);


void smartTileTilerGenerateMetaDataWithSpecifiedTileSize(std::vector<uint16_t> srcSize,
                                                         std::vector<uint16_t> outSize,
                                                         std::vector<xF::smartTileMetaData>& metaDataList,
                                                         uint16_t& numberOfTileRows,
                                                         uint16_t& numberOfTileColumns,
                                                         int tileAlignment = 1,
                                                         bool enableInvalidRegions = true);
} // namespace xF

#endif //_SMARTTILERSTITCHER_HPP_
