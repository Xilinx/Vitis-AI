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
            errorMessage << argv[0] << " <xclbin> <inputImage1> <inputImage2> "
                                       "<scale> [width] [height] [iterations]";
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

        int width = srcImageR.cols;
        if (argc >= 6) width = atoi(argv[5]);
        int height = srcImageR.rows;
        if (argc >= 7) height = atoi(argv[6]);

        if ((width != srcImageR.cols) || (height != srcImageR.rows)) {
            cv::resize(srcImageR, srcImageR, cv::Size(width, height));
        }
        if ((width != srcImageR1.cols) || (height != srcImageR1.rows)) {
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

        // Initializa device
        xF::deviceInit(xclBinName);

        // Load image
        std::vector<int16_t> srcData;
        srcData.assign(srcImageR.data, (srcImageR.data + srcImageR.total()));
        cv::Mat src(srcImageR.rows, srcImageR.cols, CV_16SC1, (void*)srcData.data());

        std::vector<int16_t> srcData1;
        srcData1.assign(srcImageR1.data, (srcImageR1.data + srcImageR1.total()));
        cv::Mat src1(srcImageR1.rows, srcImageR1.cols, CV_16SC1, (void*)srcData1.data());

        // Allocate output buffer
        std::vector<int16_t> dstData;
        dstData.assign(op_height * op_width, 0);

        cv::Mat dst(op_height, op_width, CV_16SC1, (void*)dstData.data());
        cv::Mat dstOutImage(op_height, op_width, srcImageR.type());

        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler(0, 0);
        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler1(0, 0);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> stitcher;

        std::cout << "Graph init. This does nothing because CDO in boot PDI "
                     "already configures AIE.\n";
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
            auto tiles_sz = tiler.host2aie_nb(srcData.data(), srcImageR.size(), {"gmioIn[0]"});
            tiler1.host2aie_nb(srcData1.data(), srcImageR1.size(), {"gmioIn[1]"});
            stitcher.aie2host_nb(dstData.data(), dst.size(), tiles_sz, {"gmioOut[0]"});
            // auto tiles_sz = tiler.host2aie_nb(src);
            // tiler1.host2aie_nb(src1);
            // stitcher.aie2host_nb(dst, tiles_sz);

            std::cout << "Graph run(" << (tiles_sz[0] * tiles_sz[1]) << ")\n";

            mygraph.run(tiles_sz[0] * tiles_sz[1]);

            mygraph.wait();
            tiler.wait({"gmioIn[0]"});
            tiler1.wait({"gmioIn[1]"});
            stitcher.wait({"gmioOut[0]"});

            STOP_TIMER("pixelwise_mul function")
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
