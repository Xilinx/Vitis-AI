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
    uint32_t groupNumber;
    std::vector<uint32_t> position;     //<row,column> in image
    std::vector<uint16_t> size;         //<height,width> of tile  //can be copied from cv::Size that is part of cv::Mat
                                        ////potential compile time constant
    std::vector<uint16_t> overlapSizeH; //<left,right>
    std::vector<uint16_t> overlapSizeV; //<top,bottom>

    std::vector<uint32_t> finalImageSize; // cv::Size finalImageSize;

    // parameters required to support resize
    std::vector<uint16_t> coordinate; //<row,column> in tile configuration
    std::vector<uint16_t>
        tileConfiguration; //<#tileRows,#tileCols>						//potential compile time
                           // constant

    // parameters to enable run-time monitoring if tile overlap size is large enough or enable on the fly retiling in an
    // dynamic environment.
    bool enableInvalidRegions; // flag to enable invalidRegion tracking					//only for
                               // dynamic execution
    std::vector<uint16_t>
        invalidHorizontalRegion; // <left,right>						//only for dynamic
                                 // execution
    std::vector<uint16_t>
        invalidVerticalRegion; // <top,bottom>							//only for dynamic
                               // execution

    smartTileMetaData(){};

    smartTileMetaData(unsigned int _groupNumber,
                      std::vector<uint16_t> _tileSize,
                      std::vector<uint32_t> _position,
                      std::vector<uint16_t> _coordinate,
                      std::vector<uint16_t> _overlapSizeH,
                      std::vector<uint16_t> _overlapSizeV,
                      std::vector<uint32_t> _finalImageSize,
                      std::vector<uint16_t> _tileConfiguration,
                      bool _enableInvalidRegions = true) {
        groupNumber = _groupNumber;
        size = _tileSize;
        position = _position;
        coordinate = _coordinate;
        overlapSizeH = _overlapSizeH;
        overlapSizeV = _overlapSizeV;
        finalImageSize = _finalImageSize;
        tileConfiguration = _tileConfiguration;
        enableInvalidRegions = _enableInvalidRegions;
        invalidHorizontalRegion = {0, 0};
        invalidVerticalRegion = {0, 0};
    }

    uint16_t tileWidth() const { return size[1]; }
    uint16_t tileHeight() const { return size[0]; }
    uint16_t positionV() const { return position[0]; }
    uint16_t positionH() const { return position[1]; }
    uint32_t finalWidth() const { return finalImageSize[1]; }
    uint32_t finalHeight() const { return finalImageSize[0]; }

    uint16_t overlapSizeH_left() const { return overlapSizeH[0]; }
    uint16_t overlapSizeH_right() const { return overlapSizeH[1]; }
    uint16_t overlapSizeV_top() const { return overlapSizeV[0]; }
    uint16_t overlapSizeV_bottom() const { return overlapSizeV[1]; }
};

} // namespace xF

#endif //_SMARTTILE_HPP_
