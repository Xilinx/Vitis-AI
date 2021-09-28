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

#include "xf_accumulate_config.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_img1, out_img;
    cv::Mat in_gray, in_gray1, diff;

// Reading in the images:
#if GRAY
    in_gray = cv::imread(argv[1], 0);
    in_gray1 = cv::imread(argv[2], 0);
#else
    in_gray = cv::imread(argv[1], 1);
    in_gray1 = cv::imread(argv[2], 1);
#endif

    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    if (in_gray1.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }

// Allocate memory for the output images:
#if GRAY
    cv::Mat out_gray(in_gray.rows, in_gray.cols, CV_16U);
    cv::Mat ocv_ref_32f(in_gray.rows, in_gray.cols, CV_32F);
    cv::Mat ocv_ref(in_gray.rows, in_gray.cols, CV_16U);
#else
    cv::Mat out_gray(in_gray.rows, in_gray.cols, CV_16UC3);
    cv::Mat ocv_ref_32f(in_gray.rows, in_gray.cols, CV_32FC3);
    cv::Mat ocv_ref(in_gray.rows, in_gray.cols, CV_16UC3);
#endif

    int height = in_gray.rows;
    int width = in_gray.cols;
    // OpenCV functions
    in_gray1.convertTo(ocv_ref_32f, CV_32F);
    cv::accumulate(in_gray, ocv_ref_32f, cv::noArray());
    ocv_ref_32f.convertTo(ocv_ref, CV_16U);

    // Write OpenCV reference image
    cv::imwrite("out_ocv.jpg", ocv_ref);

    // OpenCL section:
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * in_gray.channels() * sizeof(unsigned char);
    size_t image_out_size_bytes = in_gray.rows * in_gray.cols * in_gray.channels() * sizeof(unsigned short);

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

    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_accumulate");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "accumulate_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage1(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImage2(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage1,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray.data,        // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage2,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray1.data,       // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_gray.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Write output image
    cv::imwrite("out_hls.jpg", out_gray);

    // Compute absolute difference image
    cv::absdiff(ocv_ref, out_gray, diff);
    // Save the difference image
    cv::imwrite("diff.jpg", diff);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
#if GRAY
            float v = diff.at<short>(i, j);
            if (v > 0.0f) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
#else
            cv::Vec3s v = diff.at<cv::Vec3s>(i, j);
            if (v[0] > 0.0f) cnt++;
            if (minval > v[0]) minval = v[0];
            if (maxval < v[0]) maxval = v[0];

            if (v[1] > 0.0f) cnt++;
            if (minval > v[1]) minval = v[1];
            if (maxval < v[1]) maxval = v[1];

            if (v[2] > 0.0f) cnt++;
            if (minval > v[2]) minval = v[2];
            if (maxval < v[2]) maxval = v[2];
#endif
        }
    }
    float err_per = 100.0 * (float)cnt / (in_gray.rows * in_gray.cols * in_gray.channels());

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
