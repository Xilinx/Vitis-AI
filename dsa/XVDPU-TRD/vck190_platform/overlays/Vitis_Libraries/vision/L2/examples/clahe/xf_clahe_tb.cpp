/*
 * Copyright 2020 Xilinx, Inc.
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

#include "common/xf_headers.hpp"
#include "opencv2/imgproc/types_c.h"
#include "xf_clahe_config.h"
#include "xf_opencl_wrap.hpp"
#include <fstream>
int main(int argc, char** argv) {
    cv::Mat in_img;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image>\n");
        return -1;
    }

    // Read input image
    in_img = cv::imread(argv[1], 1);
    if (in_img.data == NULL) {
        std::cout << "Can't open image !!" << std::endl;
        return -1;
    }

    cv::resize(in_img, in_img, cv::Size(WIDTH / 4, HEIGHT / 4));

    int rows = in_img.rows;
    int cols = in_img.cols;

    // HLS Implementation [[
    cv::Mat yuv_image_hls;
    cv::cvtColor(in_img, yuv_image_hls, cv::COLOR_BGR2YUV);

    std::vector<cv::Mat> yuv_planes_hls(3);
    cv::split(yuv_image_hls, yuv_planes_hls);

    cv::Mat dst_hls(rows, cols, CV_8UC1);
    // apply the CLAHE algorithm to the L channel
    int clip = 3;
    int tilesY = 4;
    int tilesX = 4;
    (void)cl_kernel_mgr::registerKernel("clahe_accel", "krnl_clahe", XCLIN(yuv_planes_hls[0]), XCLOUT(dst_hls),
                                        XCLIN(rows), XCLIN(cols), XCLIN(clip), XCLIN(tilesY), XCLIN(tilesX));

    cl_kernel_mgr::exec_all();
    cl_kernel_mgr::exec_all();

    imwrite("hls_out_Y.png", dst_hls);
    //]]

    // OpenCV reference implementation [[
    cv::Mat yuv_image;
    cv::cvtColor(in_img, yuv_image, cv::COLOR_BGR2YUV);

    std::vector<cv::Mat> yuv_planes(3);
    cv::split(yuv_image, yuv_planes); // now we have the L image in yuv_planes[0]

    cv::Mat dst;
    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3, cv::Size(4, 4));
    clahe->apply(yuv_planes[0], dst);
    imwrite("ref_out_Y.png", dst);

    int err = 0;
    int maxError = 0;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            unsigned char s1 = dst.at<unsigned char>(i, j);
            unsigned char s2 = dst_hls.at<unsigned char>(i, j);
            unsigned char diff = abs(s1 - s2);
            if (diff > 3) {
                err++;
            }
            if (diff > maxError) maxError = diff;
        }
    }

    // Merge the the color planes back into an Lab image
    dst_hls.copyTo(yuv_planes_hls[0]);
    cv::merge(yuv_planes_hls, yuv_image_hls);

    // convert back to RGB
    cv::Mat image_clahe_hls;
    cv::cvtColor(yuv_image_hls, image_clahe_hls, cv::COLOR_YUV2BGR);

    // Write output image
    imwrite("hls_out.png", image_clahe_hls);

    // Merge the the color planes back into an Lab image
    dst.copyTo(yuv_planes[0]);
    cv::merge(yuv_planes, yuv_image);

    // convert back to RGB
    cv::Mat image_clahe;
    cv::cvtColor(yuv_image, image_clahe, cv::COLOR_YUV2BGR);

    // Write output image
    imwrite("ref_out.png", image_clahe);
    //]]

    std::cout << "Total number of errors = " << err << std::endl;
    std::cout << "Maximum difference = " << maxError << std::endl;
    if (err > 0) {
        std::cerr << "Test Failed " << std::endl;
        return -1;
    }

    std::cout << "Test Passed" << std::endl;

    return 0;
}
