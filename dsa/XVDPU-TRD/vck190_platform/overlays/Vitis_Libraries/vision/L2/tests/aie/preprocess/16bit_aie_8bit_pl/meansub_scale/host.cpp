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

#define PROFILE

#include <adf/adf_api/XRTConfig.h>
#include <chrono>
#include <common/xf_aie_sw_utils.hpp>
#include <common/xfcvDataMovers.h>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <xaiengine.h>
#include <xrt/experimental/xrt_kernel.h>

#include "graph.cpp"
#include "config.h"

int run_opencv_ref(cv::Mat& srcImageR, cv::Mat& dstRefImage, float mean, float scale, float gamma) {
    cv::Mat tmpImage;

    for (int r = 0; r < srcImageR.rows; r++) {
        for (int c = 0; c < srcImageR.cols; c++) {
            int temp;
            if (OP_MODE == 0)
                temp = (int)(srcImageR.data[r * srcImageR.cols + c] - mean);
            else if (OP_MODE == 1)
                temp = (int)((srcImageR.data[r * srcImageR.cols + c] - mean) * scale);
            else if (OP_MODE == 2)
                temp = (int)(((srcImageR.data[r * srcImageR.cols + c] - mean) * scale) + gamma);

            if (temp > 127)
                temp = 127;
            else if (temp < -128)
                temp = -128;

            dstRefImage.data[r * srcImageR.cols + c] = temp;
        }
    }

    return 0;
}

/*
 ******************************************************************************
 * Top level executable
 ******************************************************************************
 */

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::stringstream errorMessage;
            errorMessage << argv[0] << " <xclbin> <inputImage> [width] [height] [iterations]";
            std::cerr << errorMessage.str();
            throw std::invalid_argument(errorMessage.str());
        }

        const char* xclBinName = argv[1];
        //////////////////////////////////////////
        // Read image from file and resize
        //////////////////////////////////////////
        cv::Mat srcImageR;
        srcImageR = cv::imread(argv[2], 0);

        int width = srcImageR.cols;
        if (argc >= 4) width = atoi(argv[3]);
        int height = srcImageR.rows;
        if (argc >= 5) height = atoi(argv[4]);

        if ((width != srcImageR.cols) || (height != srcImageR.rows))
            cv::resize(srcImageR, srcImageR, cv::Size(width, height));

        int iterations = 1;
        if (argc >= 6) iterations = atoi(argv[5]);

        std::cout << "Image size" << std::endl;
        std::cout << srcImageR.rows << std::endl;
        std::cout << srcImageR.cols << std::endl;
        std::cout << srcImageR.elemSize() << std::endl;
        std::cout << "Image size (end)" << std::endl;
        int op_width = srcImageR.cols;
        int op_height = srcImageR.rows;

        // Mean Scale and Gamma values
        float mean = 72.25;
        float scale = 0.5;
        float gamma = 0.0;

        //////////////////////////////////////////
        // Run opencv reference test
        //////////////////////////////////////////
        cv::Mat dstRefImage;
        dstRefImage.create(srcImageR.rows, srcImageR.cols, CV_8SC1);
        run_opencv_ref(srcImageR, dstRefImage, mean, scale, gamma);

        // Initializa device
        xF::deviceInit(xclBinName);

        // Load image
        void* srcData = nullptr;
        xrtBufferHandle src_hndl = xrtBOAlloc(xF::gpDhdl, (srcImageR.total() * srcImageR.elemSize()), 0, 0);
        srcData = xrtBOMap(src_hndl);
        memcpy(srcData, srcImageR.data, (srcImageR.total() * srcImageR.elemSize()));

        // Allocate output buffer
        void* dstData = nullptr;
        xrtBufferHandle dst_hndl = xrtBOAlloc(xF::gpDhdl, (op_height * op_width * srcImageR.elemSize()), 0, 0);
        dstData = xrtBOMap(dst_hndl);
        cv::Mat dst(op_height, op_width, CV_8SC1, dstData);

        xF::xfcvDataMovers<xF::TILER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR> tiler(0, 0);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR> stitcher;

        std::cout << "Graph init. This does nothing because CDO in boot PDI "
                     "already configures AIE.\n";
        ppgraph.init();
        ppgraph.update(ppgraph.alpha, mean);
#if OP_MODE == 1 || OP_MODE == 2
        ppgraph.update(ppgraph.beta, scale);
#endif
#if OP_MODE == 2
        ppgraph.update(ppgraph.gamma, gamma);
#endif

        START_TIMER
        tiler.compute_metadata(srcImageR.size());
        STOP_TIMER("Meta data compute time")

        std::chrono::microseconds tt(0);
        for (int i = 0; i < iterations; i++) {
            //@{
            std::cout << "Iteration : " << (i + 1) << std::endl;
            START_TIMER
            auto tiles_sz = tiler.host2aie_nb(src_hndl, srcImageR.size());
            stitcher.aie2host_nb(dst_hndl, dst.size(), tiles_sz);

            std::cout << "Graph run(" << (tiles_sz[0] * tiles_sz[1]) << ")\n";

            ppgraph.run(tiles_sz[0] * tiles_sz[1]);

            ppgraph.wait();
            tiler.wait();
            stitcher.wait();

            STOP_TIMER("Pre-process function")
            std::cout << "Data transfer complete (Stitcher)\n";
            tt += tdiff;
            //@}

            // Analyze output {
            std::cout << "Analyzing diff\n";
            cv::Mat diff;
            cv::absdiff(dstRefImage, dst, diff);
            cv::imwrite("ref.png", dstRefImage);
            cv::imwrite("aie.png", dst);
            cv::imwrite("diff.png", diff);

            int err_count = 0;

            for (int ii = 0; ii < dst.rows; ii++) {
                for (int jj = 0; jj < dst.cols; jj++) {
                    if (diff.data[ii * dst.cols + jj] > 2) {
                        std::cout << " inp = " << (int)srcImageR.data[ii * dst.cols + jj]
                                  << " ref = " << (int)dstRefImage.data[ii * dst.cols + jj]
                                  << " aie = " << (int)dst.data[ii * dst.cols + jj] << std::endl;
                        err_count++;
                    }
                }
            }

            if (err_count > 0) {
                std::cerr << "Test failed" << std::endl;
                exit(-1);
            }
        }
        //}
        std::cout << "Test passed" << std::endl;
        std::cout << "Average time to process frame : " << (((float)tt.count() * 0.001) / (float)iterations) << " ms"
                  << std::endl;
        std::cout << "Average frames per second : " << (((float)1000000 / (float)tt.count()) * (float)iterations)
                  << " fps" << std::endl;

        return 0;
    } catch (std::exception& e) {
        const char* errorMessage = e.what();
        std::cerr << "Exception caught: " << errorMessage << std::endl;
        exit(-1);
    }
}
