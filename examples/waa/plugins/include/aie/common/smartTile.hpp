/*****************************************************************************
 * Copyright 2019-2020 Xilinx, Inc.
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
 *****************************************************************************/

#ifndef _SMARTTILE_HPP_
#define _SMARTTILE_HPP_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

namespace xF {

// smartTile metaData class declaration

class smartTileMetaData {
   public:
    uint32_t m_groupNumber;

    //<height,width> of tile
    uint16_t m_tileHeight;
    uint16_t m_tileWidth;

    //<row,column> in image
    uint32_t m_positionH;
    uint32_t m_positionV;

    //<row,column> in tile configuration
    uint16_t m_coordinate_row;
    uint16_t m_coordinate_col;

    //<left,right> overlap
    uint16_t m_overlapSizeH_left;
    uint16_t m_overlapSizeH_right;

    //<top,bottom> overlap
    uint16_t m_overlapSizeV_top;
    uint16_t m_overlapSizeV_bottom;

    //cv::Size finalImageSize
    uint16_t m_finalImageSizeHeight;
    uint16_t m_finalImageSizeWidth;

    //tile configuration
    uint16_t m_tileRows;
    uint16_t m_tileCols;

    bool m_enableInvalidRegions;

    uint16_t m_invalidHorizontalRegion_left;
    uint16_t m_invalidHorizontalRegion_right;
    uint16_t m_invalidVerticalRegion_top;
    uint16_t m_invalidVerticalRegion_bottom;

    smartTileMetaData(){};

    smartTileMetaData(unsigned int _groupNumber,
                      uint16_t _tileHeight,
                      uint16_t _tileWidth,
                      uint32_t _positionV,
                      uint32_t _positionH,
                      uint16_t _coordinate_row,
                      uint16_t _coordinate_col,
                      uint16_t _overlapSizeH_left,
                      uint16_t _overlapSizeH_right,
                      uint16_t _overlapSizeV_top,
                      uint16_t _overlapSizeV_bottom,
                      uint32_t _finalImageSizeHeight,
                      uint32_t _finalImageSizeWidth,
                      uint16_t _tileRows,
                      uint16_t _tileCols,
                      bool _enableInvalidRegions = true) {

        m_groupNumber = _groupNumber;

        m_tileHeight = _tileHeight;
        m_tileWidth = _tileWidth;

        m_positionH = _positionH;
        m_positionV = _positionV;

        m_coordinate_row = _coordinate_row;
        m_coordinate_col = _coordinate_col;

        m_overlapSizeH_left = _overlapSizeH_left;
        m_overlapSizeH_right = _overlapSizeH_right;

        m_overlapSizeV_top = _overlapSizeV_top;
        m_overlapSizeV_bottom = _overlapSizeV_bottom;

        m_finalImageSizeHeight = _finalImageSizeHeight;
        m_finalImageSizeWidth = _finalImageSizeWidth;

        m_tileRows = _tileRows;
        m_tileCols = _tileCols;

        m_enableInvalidRegions = _enableInvalidRegions;

        m_invalidHorizontalRegion_left = 0;
        m_invalidHorizontalRegion_right = 0;
        m_invalidVerticalRegion_top = 0;
        m_invalidVerticalRegion_bottom = 0;
    }

    uint16_t tileWidth() const { return m_tileWidth; }
    uint16_t tileHeight() const { return m_tileHeight; }
    uint16_t positionV() const { return m_positionV; }
    uint16_t positionH() const { return m_positionH; }
    uint32_t finalWidth() const { return m_finalImageSizeWidth; }
    uint32_t finalHeight() const { return m_finalImageSizeHeight; }

    uint16_t overlapSizeH_left() const { return m_overlapSizeH_left; }
    uint16_t overlapSizeH_right() const { return m_overlapSizeH_right; }
    uint16_t overlapSizeV_top() const { return m_overlapSizeV_top; }
    uint16_t overlapSizeV_bottom() const { return m_overlapSizeV_bottom; }
};

} // namespace xF

#endif //_SMARTTILE_HPP_
