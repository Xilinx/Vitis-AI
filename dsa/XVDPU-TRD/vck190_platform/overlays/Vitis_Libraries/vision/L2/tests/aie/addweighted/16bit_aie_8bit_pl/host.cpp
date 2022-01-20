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

int run_opencv_ref(
    cv::Mat& srcImageR1, cv::Mat& srcImageR2, float alpha, float beta, float gamma, cv::Mat& dstRefImage) {
    cv::addWeighted(srcImageR1, alpha, srcImageR2, beta, gamma, dstRefImage, -1);
    return 0;
}

int main(int argc, char** argv)

{
    try {
        if (argc < 7) {
            std::stringstream errorMessage;
            errorMessage
                << argv[0]
                << " <xclbin> <inputImage1> <inputImage2> <alpha> <beta> <gamma> [width] [height] [iterations]";
            std::cerr << errorMessage.str();
            throw std::invalid_argument(errorMessage.str());
        }

        const char* xclBinName = argv[1];
        //////////////////////////////////////////
        // Read image from file and resize
        //////////////////////////////////////////
        cv::Mat srcImageR1, srcImageR2;
        srcImageR1 = cv::imread(argv[2], 0);
        srcImageR2 = cv::imread(argv[3], 0);

        int width = srcImageR2.cols;
        if (argc >= 8) width = atoi(argv[7]);
        int height = srcImageR2.rows;
        if (argc >= 9) height = atoi(argv[8]);

        if ((width != srcImageR1.cols) || (height != srcImageR1.rows)) {
            cv::resize(srcImageR1, srcImageR1, cv::Size(width, height));
            cv::resize(srcImageR2, srcImageR2, cv::Size(width, height));
        }

        int iterations = 1;
        if (argc >= 10) iterations = atoi(argv[9]);

        std::cout << "Image size" << std::endl;
        std::cout << srcImageR1.rows << std::endl;
        std::cout << srcImageR1.cols << std::endl;
        std::cout << srcImageR1.elemSize() << std::endl;
        std::cout << "Image size (end)" << std::endl;
        int op_width = srcImageR1.cols;
        int op_height = srcImageR1.rows;

        cv::Mat dstRefImage(op_height, op_width, CV_8UC1);
        //////////////////////////////////////////
        // Run opencv reference test
        //////////////////////////////////////////
        float alpha = atof(argv[4]);
        float beta = atof(argv[5]);
        float gamma = atof(argv[6]);
        START_TIMER
        run_opencv_ref(srcImageR1, srcImageR2, alpha, beta, gamma, dstRefImage);
        STOP_TIMER("OpenCV Ref");

        // Initializa device
        xF::deviceInit(xclBinName);

        // Load image
        void* srcData_1 = nullptr;
        void* srcData_2 = nullptr;
        xrtBufferHandle src_hndl_1 = xrtBOAlloc(xF::gpDhdl, (srcImageR1.total() * srcImageR1.elemSize()), 0, 0);
        srcData_1 = xrtBOMap(src_hndl_1);
        memcpy(srcData_1, srcImageR1.data, (srcImageR1.total() * srcImageR1.elemSize()));

        xrtBufferHandle src_hndl_2 = xrtBOAlloc(xF::gpDhdl, (srcImageR2.total() * srcImageR2.elemSize()), 0, 0);
        srcData_2 = xrtBOMap(src_hndl_2);
        memcpy(srcData_2, srcImageR2.data, (srcImageR2.total() * srcImageR2.elemSize()));

        // Allocate output buffer
        void* dstData = nullptr;
        xrtBufferHandle dst_hndl = xrtBOAlloc(xF::gpDhdl, (op_height * op_width * srcImageR1.elemSize()), 0, 0);
        dstData = xrtBOMap(dst_hndl);
        cv::Mat dst(op_height, op_width, srcImageR1.type(), dstData);

        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR> tiler1(0, 0);
        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR> tiler2(0, 0);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR> stitcher;

        mygraph.init();
        mygraph.update(mygraph.alpha, alpha);
        mygraph.update(mygraph.beta, beta);
        mygraph.update(mygraph.gamma, gamma);

        START_TIMER
        tiler1.compute_metadata(srcImageR1.size());
        tiler2.compute_metadata(srcImageR2.size());
        STOP_TIMER("Meta data compute time")

        std::chrono::microseconds tt(0);
        for (int i = 0; i < iterations; i++) {
            //@{
            std::cout << "Iteration : " << (i + 1) << std::endl;

            START_TIMER
            auto tiles_sz = tiler1.host2aie_nb(src_hndl_1, srcImageR1.size());
            auto tiles_sz2 = tiler2.host2aie_nb(src_hndl_2, srcImageR2.size());
            stitcher.aie2host_nb(dst_hndl, dst.size(), tiles_sz);

            std::cout << "Graph run(" << (tiles_sz[0] * tiles_sz[1]) << ")\n";
            mygraph.run(tiles_sz[0] * tiles_sz[1]);
            mygraph.wait();
            std::cout << "Graph run complete\n";

            tiler1.wait();
            tiler2.wait();

            stitcher.wait();
            STOP_TIMER("addweighted function")
            tt += tdiff;
            //@}

            // Analyze output {
            std::cout << "Analyzing diff\n";
            cv::Mat diff(op_height, op_width, srcImageR1.type());
            cv::absdiff(dstRefImage, dst, diff);
            cv::imwrite("ref.png", dstRefImage);
            cv::imwrite("aie.png", dst);
            cv::imwrite("diff.png", diff);

            float err_per;
            analyzeDiff(diff, 1, err_per);
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
