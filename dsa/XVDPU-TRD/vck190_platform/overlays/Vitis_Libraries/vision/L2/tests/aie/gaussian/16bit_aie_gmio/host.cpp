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

int run_opencv_ref(cv::Mat& srcImageR, cv::Mat& dstRefImage) {
    float sigma = 0.5f;
    cv::GaussianBlur(srcImageR, dstRefImage, cv::Size(3, 3), sigma, sigma, cv::BORDER_REPLICATE);
    return 0;
}

/*
 ******************************************************************************
 * Top level executable
 ******************************************************************************
 */

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::stringstream errorMessage;
            errorMessage << argv[0] << " <xclbin> <inputImage> [width] [height] [iterations] ";
            std::cerr << errorMessage.str();
            throw std::invalid_argument(errorMessage.str());
        }

        const char* xclBinName = argv[1];
        //////////////////////////////////////////
        // Read image from file and resize
        //////////////////////////////////////////
        cv::Mat srcImage;
        srcImage = cv::imread(argv[2], 0);

        int width = srcImage.cols;
        if (argc >= 4) width = atoi(argv[3]);
        int height = srcImage.rows;
        if (argc >= 5) height = atoi(argv[4]);

        if ((width != srcImage.cols) || (height != srcImage.rows))
            cv::resize(srcImage, srcImage, cv::Size(width, height));

        int iterations = 1;
        if (argc >= 6) iterations = atoi(argv[5]);

        std::cout << "Image size" << std::endl;
        std::cout << srcImage.rows << std::endl;
        std::cout << srcImage.cols << std::endl;
        std::cout << srcImage.elemSize() << std::endl;
        std::cout << "Image size (end)" << std::endl;
        int op_width = srcImage.cols;
        int op_height = srcImage.rows;

        //////////////////////////////////////////
        // Run opencv reference test (gaussian design)
        //////////////////////////////////////////
        cv::Mat dstRefImage;
        run_opencv_ref(srcImage, dstRefImage);

        // Initializa device
        xF::deviceInit(xclBinName);

        // Load image
        std::vector<int16_t> srcData;
        srcData.assign(srcImage.data, (srcImage.data + srcImage.total()));
        cv::Mat src(op_height, op_width, CV_16SC1, (void*)srcData.data());

        // Allocate output buffer
        std::vector<int16_t> dstData;
        dstData.assign(op_width * op_height, 0);
        cv::Mat dst(op_height, op_width, CV_16SC1, (void*)dstData.data());
        cv::Mat dstOutImage(op_height, op_width, srcImage.type());

        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler(1, 1);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> stitcher;

        std::cout << "Graph init. This does nothing because CDO in boot PDI already configures AIE.\n";
        gaussian_graph.init();
        gaussian_graph.update(gaussian_graph.kernelCoefficients, float2fixed_coeff<10, 16>(kData).data(), 16);

        START_TIMER
        tiler.compute_metadata(srcImage.size());
        STOP_TIMER("Meta data compute time")

        std::chrono::microseconds tt(0);
        for (int i = 0; i < iterations; i++) {
            std::cout << "Iteration : " << (i + 1) << std::endl;
            //@{
            START_TIMER
            auto tiles_sz = tiler.host2aie_nb(srcData.data(), srcImage.size(), {"gmioIn[0]"});
            stitcher.aie2host_nb(dstData.data(), dst.size(), tiles_sz, {"gmioOut[0]"});

            gaussian_graph.run(tiles_sz[0] * tiles_sz[1]);
            gaussian_graph.wait();

            tiler.wait({"gmioIn[0]"});

            stitcher.wait({"gmioOut[0]"});
            STOP_TIMER("Total time to process frame")
            tt += tdiff;
            //@}

            // Analyze output {
            std::cout << "Analyzing diff\n";
            cv::Mat diff;
            dst = cv::max(dst, 0);
            dst = cv::min(dst, 255);
            dst.convertTo(dstOutImage, srcImage.type());
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
        std::cout << "Test passed" << std::endl;
        std::cout << "Average time to process frame : " << (((float)tt.count() * 0.001) / (float)iterations) << " ms"
                  << std::endl;
        //}
        std::cout << "Test passed" << std::endl;
        return 0;
    } catch (std::exception& e) {
        const char* errorMessage = e.what();
        std::cerr << "Exception caught: " << errorMessage << std::endl;
        exit(-1);
    }
}
