/*
 * Copyright 2021 Xilinx, Inc.
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

} // namespace xF

#endif //_SMARTTILERSTITCHER_HPP_
