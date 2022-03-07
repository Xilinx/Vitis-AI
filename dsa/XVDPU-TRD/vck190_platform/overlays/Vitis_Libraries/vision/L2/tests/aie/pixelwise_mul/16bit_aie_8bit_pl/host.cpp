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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xaiengine.h>

#include "graph.cpp"

int run_opencv_ref(cv::Mat& srcImageR, cv::Mat& srcImageR1, cv::Mat& dstRefImage, float scale) {
    cv::multiply(srcImageR, srcImageR1, dstRefImage, scale, srcImageR.type());
    return 0;
}

/*
 ******************************************************************************
 * Top level executable
 ******************************************************************************
 */

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::stringstream errorMessage;
            errorMessage << argv[0] << " <xclbin> <inputImage1> <inputImage2> <scale> [width] [height] [iterations]";
            std::cerr << errorMessage.str();
            throw std::invalid_argument(errorMessage.str());
        }

        const char* xclBinName = argv[1];
        //////////////////////////////////////////
        // Read image from file and resize
        //////////////////////////////////////////
        cv::Mat srcImageR, srcImageR1;
        srcImageR = cv::imread(argv[2], 0);
        srcImageR1 = cv::imread(argv[3], 0);

        int width = srcImageR1.cols;
        if (argc >= 6) width = atoi(argv[5]);
        int height = srcImageR1.rows;
        if (argc >= 7) height = atoi(argv[6]);

        if ((width != srcImageR.cols) || (height != srcImageR.rows)) {
            cv::resize(srcImageR, srcImageR, cv::Size(width, height));
            cv::resize(srcImageR1, srcImageR1, cv::Size(width, height));
        }

        int iterations = 1;
        if (argc >= 8) iterations = atoi(argv[7]);

        std::cout << "Image size" << std::endl;
        std::cout << srcImageR.rows << std::endl;
        std::cout << srcImageR.cols << std::endl;
        std::cout << srcImageR.elemSize() << std::endl;
        std::cout << "Image size (end)" << std::endl;
        int op_width = srcImageR.cols;
        int op_height = srcImageR.rows;

        float scale = atof(argv[4]);
        //////////////////////////////////////////
        // Run opencv reference test (filter2D design)
        //////////////////////////////////////////
        cv::Mat dstRefImage;
        run_opencv_ref(srcImageR, srcImageR1, dstRefImage, scale);

        // Initialize device
        xF::deviceInit(xclBinName);

        // Load image
        void* srcData = nullptr;
        void* srcData1 = nullptr;
        xrtBufferHandle src_hndl = xrtBOAlloc(xF::gpDhdl, (srcImageR.total() * srcImageR.elemSize()), 0, 0);
        xrtBufferHandle src_hndl1 = xrtBOAlloc(xF::gpDhdl, (srcImageR1.total() * srcImageR1.elemSize()), 0, 0);
        srcData = xrtBOMap(src_hndl);
        srcData1 = xrtBOMap(src_hndl1);
        memcpy(srcData, srcImageR.data, (srcImageR.total() * srcImageR.elemSize()));
        memcpy(srcData1, srcImageR1.data, (srcImageR1.total() * srcImageR1.elemSize()));

        // Allocate output buffer
        void* dstData = nullptr;
        xrtBufferHandle dst_hndl = xrtBOAlloc(xF::gpDhdl, (op_height * op_width * srcImageR.elemSize()), 0, 0);
        dstData = xrtBOMap(dst_hndl);
        cv::Mat dst(op_height, op_width, srcImageR.type(), dstData);

        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR> tiler(0, 0);
        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR> tiler1(0, 0);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR> stitcher;

        std::cout << "Graph init. This does nothing because CDO in boot PDI already configures AIE.\n";
        mygraph.init();
        mygraph.update(mygraph.scale, scale);

        START_TIMER
        tiler.compute_metadata(srcImageR.size());
        tiler1.compute_metadata(srcImageR1.size());
        STOP_TIMER("Meta data compute time")

        std::chrono::microseconds tt(0);
        for (int i = 0; i < iterations; i++) {
            //@{
            std::cout << "Iteration : " << (i + 1) << std::endl;
            START_TIMER
            auto tiles_sz = tiler.host2aie_nb(src_hndl, srcImageR.size());
            tiler1.host2aie_nb(src_hndl1, srcImageR1.size());
            stitcher.aie2host_nb(dst_hndl, dst.size(), tiles_sz);

            std::cout << "Graph run(" << (tiles_sz[0] * tiles_sz[1]) << ")\n";
            mygraph.run(tiles_sz[0] * tiles_sz[1]);
            mygraph.wait();
            std::cout << "Graph run complete\n";

            tiler.wait();
            tiler1.wait();
            stitcher.wait();

            STOP_TIMER("pixelwise_mul function")
            tt += tdiff;
            //@}

            // Analyze output {
            std::cout << "Analyzing diff\n";
            cv::Mat diff;
            cv::absdiff(dstRefImage, dst, diff);
            cv::imwrite("ref.png", dstRefImage);
            cv::imwrite("aie.png", dst);
            cv::imwrite("diff.png", diff);

            float err_per;
            analyzeDiff(diff, 2, err_per);
            if (err_per > 0.0f) {
                std::cerr << "Test failed" << std::endl;
                exit(-1);
            }
            //}
        }
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
