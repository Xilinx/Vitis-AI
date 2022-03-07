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

int run_opencv_ref(cv::Mat& srcImageR, cv::Mat& dstRefImage, float coeff[9]) {
    cv::Mat tmpImage;
    cv::Mat kernel = cv::Mat(3, 3, CV_32F, coeff);
    cv::filter2D(srcImageR, dstRefImage, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
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

        //////////////////////////////////////////
        // Run opencv reference test (filter2D design)
        //////////////////////////////////////////
        cv::Mat dstRefImage;
        run_opencv_ref(srcImageR, dstRefImage, kData);

        // Initializa device
        xF::deviceInit(xclBinName);

        // Load image

        std::vector<int16_t> srcData;
        srcData.assign(srcImageR.data, (srcImageR.data + srcImageR.total()));
        cv::Mat src(srcImageR.rows, srcImageR.cols, CV_16SC1, (void*)srcData.data());

        // Allocate output buffer
        std::vector<int16_t> dstData;
        dstData.assign(op_height * op_width, 0);

        cv::Mat dst(op_height, op_width, CV_16SC1, (void*)dstData.data());
        cv::Mat dstOutImage(op_height, op_width, srcImageR.type());

        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler(1, 1);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> stitcher;

        std::cout << "Graph init. This does nothing because CDO in boot PDI "
                     "already configures AIE.\n";
        filter_graph.init();
        filter_graph.update(filter_graph.kernelCoefficients, float2fixed_coeff<10, 16>(kData).data(), 16);

        START_TIMER
        tiler.compute_metadata(srcImageR.size());
        STOP_TIMER("Meta data compute time")

        std::chrono::microseconds tt(0);
        for (int i = 0; i < iterations; i++) {
            //@{
            std::cout << "Iteration : " << (i + 1) << std::endl;
            START_TIMER
            auto tiles_sz = tiler.host2aie_nb(srcData.data(), srcImageR.size(), {"gmioIn[0]"});
            stitcher.aie2host_nb(dstData.data(), dst.size(), tiles_sz, {"gmioOut[0]"});

            std::cout << "Graph run(" << (tiles_sz[0] * tiles_sz[1]) << ")\n";

            filter_graph.run(tiles_sz[0] * tiles_sz[1]);

            filter_graph.wait();
            tiler.wait({"gmioIn[0]"});
            stitcher.wait({"gmioOut[0]"});

            STOP_TIMER("filter2D function")
            tt += tdiff;
            //@}

            // Saturate the output values to [0,255]
            dst = cv::max(dst, 0);
            dst = cv::min(dst, 255);

            // Convert 16-bit output to 8-bit
            dst.convertTo(dstOutImage, srcImageR.type());

            // Analyze output {
            std::cout << "Analyzing diff\n";
            cv::Mat diff;
            cv::absdiff(dstRefImage, dstOutImage, diff);
            cv::imwrite("ref.png", dstRefImage);
            cv::imwrite("aie.png", dstOutImage);
            cv::imwrite("diff.png", diff);

            float err_per;
            analyzeDiff(diff, 2, err_per);
            if (err_per > 0.0f) {
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
