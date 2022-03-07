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

int run_opencv_ref(cv::Mat& srcImage1, cv::Mat& srcImage2, cv::Mat& dstRefImage, float alpha) {
    dstRefImage.create(srcImage1.rows, srcImage1.cols, CV_8UC1);
    cv::Mat ocv_ref_in1(srcImage1.rows, srcImage1.cols, CV_32FC1, 1);
    cv::Mat ocv_ref_in2(srcImage1.rows, srcImage1.cols, CV_32FC1, 1);

    srcImage1.convertTo(ocv_ref_in1, CV_32FC1);
    srcImage2.convertTo(ocv_ref_in2, CV_32FC1);

    // OpenCV function
    cv::accumulateWeighted(ocv_ref_in1, ocv_ref_in2, alpha, cv::noArray());
    ocv_ref_in2.convertTo(dstRefImage, CV_8U);

    return 0;
}

/*
 ******************************************************************************
 * Top level executable
 ******************************************************************************
 */

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::stringstream errorMessage;
            errorMessage << argv[0] << " <xclbin> <inputImage1> <inputImage2> <alpha> ";
            std::cerr << errorMessage.str();
            throw std::invalid_argument(errorMessage.str());
        }

        const char* xclBinName = argv[1];
        //////////////////////////////////////////
        // Read image from file and resize
        //////////////////////////////////////////
        cv::Mat srcImage1, srcImage2;
        srcImage1 = cv::imread(argv[2], 0);
        srcImage2 = cv::imread(argv[3], 0);
        std::cout << "Image1 size" << std::endl;
        std::cout << srcImage1.rows << std::endl;
        std::cout << srcImage1.cols << std::endl;
        std::cout << srcImage1.elemSize() << std::endl;
        std::cout << "Image2 size" << std::endl;
        std::cout << srcImage2.rows << std::endl;
        std::cout << srcImage2.cols << std::endl;
        std::cout << srcImage2.elemSize() << std::endl;
        std::cout << "Image size (end)" << std::endl;
        int op_width = srcImage1.cols;
        int op_height = srcImage1.rows;

        //////////////////////////////////////////
        // Run opencv reference test (absdiff design)
        //////////////////////////////////////////
        cv::Mat dstRefImage;
        float alpha = atof(argv[4]);
        run_opencv_ref(srcImage1, srcImage2, dstRefImage, alpha);

        // Initializa device
        xF::deviceInit(xclBinName);

        // Load image
        std::vector<int16_t> srcData1;
        srcData1.assign(srcImage1.data, (srcImage1.data + srcImage1.total()));
        cv::Mat src1(srcImage1.rows, srcImage1.cols, CV_16SC1, (void*)srcData1.data());

        std::vector<int16_t> srcData2;
        srcData2.assign(srcImage2.data, (srcImage2.data + srcImage2.total()));
        cv::Mat src2(srcImage2.rows, srcImage2.cols, CV_16SC1, (void*)srcData2.data());

        // Allocate output buffer
        std::vector<int16_t> dstData;
        dstData.assign(srcImage2.rows * srcImage2.cols, 0);
        cv::Mat dst(op_height, op_width, CV_16SC1, (void*)dstData.data());
        cv::Mat dstOutImage(op_height, op_width, srcImage1.type());

        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler1(0, 0);
        xF::xfcvDataMovers<xF::TILER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler2(0, 0);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, TILE_HEIGHT, TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> stitcher;

        std::cout << "Graph init. This does nothing because CDO in boot PDI already configures AIE.\n";
        accumw_graph.init();
        accumw_graph.update(accumw_graph.alpha, alpha);

        START_TIMER
        tiler1.compute_metadata(srcImage1.size());
        tiler2.compute_metadata(srcImage1.size());
        STOP_TIMER("Meta data compute time")

        //@{
        START_TIMER
        auto tiles_sz = tiler1.host2aie_nb(srcData1.data(), srcImage1.size(), {"gmioIn[0]"});
        tiler2.host2aie_nb(srcData2.data(), srcImage2.size(), {"gmioIn[1]"});
        stitcher.aie2host_nb(dstData.data(), dst.size(), tiles_sz, {"gmioOut[0]"});

        accumw_graph.run(tiles_sz[0] * tiles_sz[1]);
        accumw_graph.wait();

        tiler1.wait({"gmioIn[0]"});
        tiler2.wait({"gmioIn[1]"});

        stitcher.wait({"gmioOut[0]"});
        STOP_TIMER("Total time to process frame")
        std::cout << "Data transfer complete (Stitcher)\n";
        //@}

        std::cout << "Analyzing diff\n";
        cv::Mat diff;
        dst = cv::max(dst, 0);
        dst = cv::min(dst, 255);
        dst.convertTo(dstOutImage, srcImage1.type());
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
        std::cout << "Test passed" << std::endl;
        return 0;
    } catch (std::exception& e) {
        const char* errorMessage = e.what();
        std::cerr << "Exception caught: " << errorMessage << std::endl;
        exit(-1);
    }
}
