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
#include "xf_accumulate_weighted_config.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_img1, out_img;
    cv::Mat in_gray, in_gray1, diff;
#if GRAY
    in_gray = cv::imread(argv[1], 0);  // read image
    in_gray1 = cv::imread(argv[2], 0); // read image
#else
    in_gray = cv::imread(argv[1], 1);  // read image
    in_gray1 = cv::imread(argv[2], 1); // read image
#endif
    int height = in_gray.rows;
    int width = in_gray.cols;
    if (in_gray.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[1]);
        return -1;
    }
    if (in_gray1.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[2]);
        return -1;
    }
#if GRAY
    cv::Mat inout_gray(in_gray.rows, in_gray.cols, CV_16U, 1);
    cv::Mat out_gray(in_gray.rows, in_gray.cols, CV_16U, 1);
    cv::Mat inout_gray1(in_gray.rows, in_gray.cols, CV_32FC1, 1);

    cv::Mat ocv_ref(in_gray.rows, in_gray.cols, CV_16U, 1);
    cv::Mat ocv_ref_in1(in_gray.rows, in_gray.cols, CV_32FC1, 1);
    cv::Mat ocv_ref_in2(in_gray.rows, in_gray.cols, CV_32FC1, 1);

    in_gray.convertTo(ocv_ref_in1, CV_32FC1);
    in_gray1.convertTo(ocv_ref_in2, CV_32FC1);
#else
    cv::Mat inout_gray(in_gray.rows, in_gray.cols, CV_16UC3, 1);
    cv::Mat out_gray(in_gray.rows, in_gray.cols, CV_16UC3, 1);
    cv::Mat inout_gray1(in_gray.rows, in_gray.cols, CV_32FC3, 1);

    cv::Mat ocv_ref(in_gray.rows, in_gray.cols, CV_16UC3, 1);
    cv::Mat ocv_ref_in1(in_gray.rows, in_gray.cols, CV_32FC3, 1);
    cv::Mat ocv_ref_in2(in_gray.rows, in_gray.cols, CV_32FC3, 1);

    in_gray.convertTo(ocv_ref_in1, CV_32FC3);
    in_gray1.convertTo(ocv_ref_in2, CV_32FC3);

#endif
    // Weight ( 0 to 1 )
    float alpha = 0.76;

    // OpenCV function
    cv::accumulateWeighted(ocv_ref_in1, ocv_ref_in2, alpha, cv::noArray());
#if GRAY
    ocv_ref_in2.convertTo(ocv_ref, CV_16U);
#else
    ocv_ref_in2.convertTo(ocv_ref, CV_16UC3);
#endif
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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_accumulateweighted");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "accumulateweighted", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage1(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImage2(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
    OCL_CHECK(err, err = kernel.setArg(2, alpha));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(4, height));
    OCL_CHECK(err, err = kernel.setArg(5, width));

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

    cv::imwrite("out_hls.jpg", out_gray);
#if GRAY
    out_gray.convertTo(inout_gray1, CV_32FC1);
#else
    out_gray.convertTo(inout_gray1, CV_32FC3);
#endif

    // Compute absolute difference image
    absdiff(ocv_ref_in2, inout_gray1, diff);

    // Save the difference image
    cv::imwrite("diff.png", diff);

    // Find minimum and maximum differences
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
            float v = diff.at<float>(i, j);
            if (v > 1) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_gray.rows * in_gray.cols);

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
