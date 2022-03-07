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

#include "common/xf_headers.hpp"
#include <stdlib.h>
#include <ap_int.h>

#include "xf_gtm_config.h"
#include "xcl2.hpp"
#include "xf_opencl_wrap.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <INPUT IMAGE PATH > \n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat hdr_img, in_xyz, out_img, out_hls, matlab_y, diff;
    cv::Mat xyzchannel[3], _xyzchannel[3];

    // Reading in the images:
    hdr_img = cv::imread(argv[1], -1);

    if (hdr_img.data == NULL) {
        printf("ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    diff.create(hdr_img.rows, hdr_img.cols, CV_8UC1);

    in_xyz.create(hdr_img.rows, hdr_img.cols, CV_16UC3);
    out_img.create(hdr_img.rows, hdr_img.cols, CV_8UC3);
    out_hls.create(hdr_img.rows, hdr_img.cols, CV_8UC3);

    int height = hdr_img.rows;
    int width = hdr_img.cols;

    cv::cvtColor(hdr_img, in_xyz, cv::COLOR_BGR2XYZ);
    cv::split(in_xyz, xyzchannel);

    _xyzchannel[0].create(hdr_img.rows, hdr_img.cols, CV_8UC1);
    _xyzchannel[1].create(hdr_img.rows, hdr_img.cols, CV_8UC1);
    _xyzchannel[2].create(hdr_img.rows, hdr_img.cols, CV_8UC1);

    float c1 = 3.0;
    float c2 = 1.5;

    float maxL = 0, minL = 100;
    float mean = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float pxl_val = log10(xyzchannel[1].at<ushort>(i, j));
            mean = mean + pxl_val;
            maxL = (maxL > pxl_val) ? maxL : pxl_val;
            minL = (minL < pxl_val) ? minL : pxl_val;
        }
    }
    mean = mean / (height * width);

    double maxLd, minLd;
    maxLd = 2.4;
    minLd = 0;

    float K1 = (maxLd - minLd) / (maxL - minL);
    float K2;

    float d0 = maxL - minL;
    float sigma_sq = (c1 * c1) / (2 * d0 * d0);

    float val, out_val;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            val = log10(xyzchannel[1].at<ushort>(i, j));

            K2 = (1 - K1) * exp(-((val - mean) * (val - mean)) * sigma_sq) + K1;
            out_val = exp(c2 * K2 * (val - mean) + mean);

            int x_val = xyzchannel[0].at<ushort>(i, j);
            int y_val = xyzchannel[1].at<ushort>(i, j);
            int z_val = xyzchannel[2].at<ushort>(i, j);

            _xyzchannel[0].at<uchar>(i, j) = (uint8_t)((out_val / y_val) * x_val);
            _xyzchannel[2].at<uchar>(i, j) = (uint8_t)((out_val / y_val) * z_val);
            _xyzchannel[1].at<uchar>(i, j) = (uint8_t)out_val;
        }
    }
    cv::Mat out_xyz;
    cv::merge(_xyzchannel, 3, out_xyz);
    cv::cvtColor(out_xyz, out_img, cv::COLOR_XYZ2BGR);

    // CL section
    size_t image_in_size_bytes = hdr_img.rows * hdr_img.cols * 3 * sizeof(unsigned short);
    size_t image_out_size_bytes = hdr_img.rows * hdr_img.cols * 3 * sizeof(unsigned char);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_gtm");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "gtm_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    std::cout << "kernel args" << std::endl;
    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(2, c1));
    OCL_CHECK(err, err = kernel.setArg(3, c2));
    OCL_CHECK(err, err = kernel.setArg(4, hdr_img.rows));
    OCL_CHECK(err, err = kernel.setArg(5, hdr_img.cols));

    for (int i = 0; i < 2; i++) {
        // Initialize the buffers:
        cl_ulong start = 0;
        cl_ulong end = 0;
        double diff_prof = 0.0f;
        cl::Event event;

        OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                                CL_TRUE,             // blocking call
                                                0,                   // buffer offset in bytes
                                                image_in_size_bytes, // Size in bytes
                                                hdr_img.data,        // Pointer to the data to copy
                                                nullptr, &event));

        std::cout << "before enqueue task" << std::endl;

        // Execute the kernel:
        OCL_CHECK(err, err = queue.enqueueTask(kernel));

        std::cout << "after enqueue task" << std::endl;

        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        diff_prof = end - start;
        std::cout << "Kernel execution time: " << (diff_prof / 1000000) << "ms" << std::endl;

        // Copy Result from Device Global Memory to Host Local Memory
        queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                                CL_TRUE,         // blocking call
                                0,               // offset
                                image_out_size_bytes,
                                out_hls.data, // Data will be stored here
                                nullptr, &event);
    }

    // Clean up:
    queue.finish();

    std::cout << "after finish" << std::endl;

    imwrite("out_img.jpg", out_img);
    imwrite("out_hls.jpg", out_hls);

    // Compute absolute difference image
    cv::absdiff(out_img, out_hls, diff);

    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    return 0;
}