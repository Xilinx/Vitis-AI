/*
 * Copyright 2019 Xilinx, Inc.
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
#include "xcl2.hpp"
#include "xf_stereolbm_config.h"

#include <time.h>

using namespace std;

#define _TEXTURE_THRESHOLD_ 20
#define _UNIQUENESS_RATIO_ 15
#define _PRE_FILTER_CAP_ 31
#define _MIN_DISP_ 0

int main(int argc, char** argv) {
    cv::setUseOptimized(false);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat left_img, right_img;

    // Reading in the images: Only Grayscale image
    left_img = cv::imread(argv[1], 0);
    right_img = cv::imread(argv[2], 0);

    if (left_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    if (right_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }

    int rows = left_img.rows;
    int cols = left_img.cols;

    cv::Mat disp, hls_disp;

    // OpenCV reference function:
    /*    cv::StereoBM bm;
        bm.state->preFilterCap = _PRE_FILTER_CAP_;
        bm.state->preFilterType = CV_STEREO_BM_XSOBEL;
        bm.state->SADWindowSize = SAD_WINDOW_SIZE;
        bm.state->minDisparity = _MIN_DISP_;
        bm.state->numberOfDisparities = NO_OF_DISPARITIES;
        bm.state->textureThreshold = _TEXTURE_THRESHOLD_;
        bm.state->uniquenessRatio = _UNIQUENESS_RATIO_;
        bm(left_img, right_img, disp);
    */
    // enable this if the above code is obsolete
    cv::Ptr<cv::StereoBM> stereobm = cv::StereoBM::create(NO_OF_DISPARITIES, SAD_WINDOW_SIZE);
    stereobm->setPreFilterCap(_PRE_FILTER_CAP_);
    stereobm->setUniquenessRatio(_UNIQUENESS_RATIO_);
    stereobm->setTextureThreshold(_TEXTURE_THRESHOLD_);

    // TIMER START CODE
    struct timespec begin_hw, end_hw;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    stereobm->compute(left_img, right_img, disp);

    // TIMER END CODE
    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;
    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

    cv::Mat disp8, hls_disp8;
    disp.convertTo(disp8, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
    cv::imwrite("ocv_output.png", disp8);
    // end of reference

    // Creating host memory for the hw acceleration
    hls_disp.create(rows, cols, CV_16UC1);
    hls_disp8.create(rows, cols, CV_8UC1);

    // OpenCL section:
    std::vector<unsigned char> bm_state_params(4);
    bm_state_params[0] = _PRE_FILTER_CAP_;
    bm_state_params[1] = _UNIQUENESS_RATIO_;
    bm_state_params[2] = _TEXTURE_THRESHOLD_;
    bm_state_params[3] = _MIN_DISP_;

    size_t image_in_size_bytes = rows * cols * sizeof(unsigned char);
    size_t vec_in_size_bytes = bm_state_params.size() * sizeof(unsigned char);
    size_t image_out_size_bytes = rows * cols * sizeof(unsigned short int);

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_stereolbm");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "stereolbm_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImageL(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImageR(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inVecBM(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImageL));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImageR));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inVecBM));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(4, rows));
    OCL_CHECK(err, err = kernel.setArg(5, cols));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImageL,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            left_img.data,       // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImageR,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            right_img.data,      // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inVecBM,         // buffer on the FPGA
                                            CL_TRUE,                // blocking call
                                            0,                      // buffer offset in bytes
                                            vec_in_size_bytes,      // Size in bytes
                                            bm_state_params.data(), // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            hls_disp.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Convert 16U output to 8U output:
    hls_disp.convertTo(hls_disp8, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
    cv::imwrite("hls_out.jpg", hls_disp8);

    ////////  FUNCTIONAL VALIDATION  ////////
    // changing the invalid value from negative to zero for validating the
    // difference
    cv::Mat disp_u(rows, cols, CV_16UC1);
    for (int i = 0; i < disp.rows; i++) {
        for (int j = 0; j < disp.cols; j++) {
            if (disp.at<short>(i, j) < 0) {
                disp_u.at<unsigned short>(i, j) = 0;
            } else
                disp_u.at<unsigned short>(i, j) = (unsigned short)disp.at<short>(i, j);
        }
    }

    cv::Mat diff;
    diff.create(left_img.rows, left_img.cols, CV_16UC1);
    cv::absdiff(disp_u, hls_disp, diff);
    cv::imwrite("diff_img.jpg", diff);

    // removing border before diff analysis
    cv::Mat diff_c;
    diff_c.create((diff.rows - SAD_WINDOW_SIZE << 1), diff.cols - (SAD_WINDOW_SIZE << 1), CV_16UC1);
    cv::Rect roi;
    roi.x = SAD_WINDOW_SIZE;
    roi.y = SAD_WINDOW_SIZE;
    roi.width = diff.cols - (SAD_WINDOW_SIZE << 1);
    roi.height = diff.rows - (SAD_WINDOW_SIZE << 1);
    diff_c = diff(roi);

    float err_per;
    xf::cv::analyzeDiff(diff_c, 0, err_per);
    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return 1;
    }
    std::cout << "Test Passed" << std::endl;

    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is " << hw_time << "ms" << std::endl;

    return 0;
}
