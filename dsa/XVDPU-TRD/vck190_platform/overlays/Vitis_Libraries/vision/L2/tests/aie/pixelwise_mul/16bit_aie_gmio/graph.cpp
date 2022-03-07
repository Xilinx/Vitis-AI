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

#include "graph.h"

GMIO gmioIn[2] = {GMIO("gmioIn1", 256, 1000), GMIO("gmioIn2", 256, 1000)};
GMIO gmioOut[1] = {GMIO("gmioOut1", 256, 1000)};

// connect dataflow graph to simulation platform
simulation::platform<2, 1> platform(&gmioIn[0], &gmioIn[1], &gmioOut[0]);

// instantiate adf dataflow graph to compute weighted moving average
pixelwiseMulGraph mygraph;

connect<> net0(platform.src[0], mygraph.in1);
connect<> net1(platform.src[1], mygraph.in2);
connect<> net2(mygraph.out, platform.sink[0]);

// initialize and run the dataflow graph
#if defined(__AIESIM__) || defined(__X86SIM__)
#include <common/xf_aie_utils.hpp>
int main(int argc, char** argv) {
    int BLOCK_SIZE_in_Bytes = TILE_WINDOW_SIZE;

    int16_t* inputData = (int16_t*)GMIO::malloc(BLOCK_SIZE_in_Bytes);
    int16_t* inputData1 = (int16_t*)GMIO::malloc(BLOCK_SIZE_in_Bytes);
    int16_t* outputData = (int16_t*)GMIO::malloc(BLOCK_SIZE_in_Bytes);

    memset(inputData, 0, BLOCK_SIZE_in_Bytes);
    memset(inputData1, 0, BLOCK_SIZE_in_Bytes);

    xf::cv::aie::xfSetTileWidth(inputData, TILE_WIDTH);
    xf::cv::aie::xfSetTileHeight(inputData, TILE_HEIGHT);
    xf::cv::aie::xfSetTileWidth(inputData1, TILE_WIDTH);
    xf::cv::aie::xfSetTileHeight(inputData1, TILE_HEIGHT);

    int16_t* dataIn = (int16_t*)xf::cv::aie::xfGetImgDataPtr(inputData);
    int16_t* dataIn1 = (int16_t*)xf::cv::aie::xfGetImgDataPtr(inputData1);
    for (int i = 0; i < TILE_ELEMENTS; i++) {
        dataIn[i] = rand() % 256;
        dataIn1[i] = rand() % 256;
    }

    float scale = 0.05;
    mygraph.init();
    mygraph.update(mygraph.scale, scale);
    mygraph.run(1);

    gmioIn[0].gm2aie_nb(inputData, BLOCK_SIZE_in_Bytes);
    gmioIn[1].gm2aie_nb(inputData1, BLOCK_SIZE_in_Bytes);
    gmioOut[0].aie2gm_nb(outputData, BLOCK_SIZE_in_Bytes);
    gmioOut[0].wait();

    // Compare the results
    int acceptableError = 1;
    int errCount = 0;
    int16_t* dataOut = (int16_t*)xf::cv::aie::xfGetImgDataPtr(outputData);
    for (int i = 0; i < TILE_ELEMENTS; i++) {
        int cValue = (dataIn[i] * dataIn1[i] * scale);
        if (abs(dataOut[i] - cValue) > acceptableError) errCount++;
    }
    if (errCount) {
        std::cout << "Test failed!" << std::endl;
        exit(-1);
    }
    std::cout << "Test passed!" << std::endl;

    mygraph.end();

    return 0;
}
#endif
