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
#include "xf_gaussian_diff_config.h"

#include <time.h>

using namespace cv;

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat out_img, ocv_ref, in_gray, diff;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (!in_gray.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Create memory for output image
    ocv_ref.create(in_gray.rows, in_gray.cols, in_gray.depth());
    out_img.create(in_gray.rows, in_gray.cols, in_gray.depth());

#if FILTER_WIDTH == 3
    float sigma = 0.5f;
#endif
#if FILTER_WIDTH == 7
    float sigma = 1.16666f;
#endif
#if FILTER_WIDTH == 5
    float sigma = 0.8333f;
#endif

    cv::Mat dst(in_gray.size(), in_gray.type());
    cv::Mat dst2(in_gray.size(), in_gray.type());
    cv::Mat dst3(in_gray.size(), in_gray.type());
    cv::Mat dst4(in_gray.size(), in_gray.type());
    cv::Mat dst_fin(in_gray.size(), in_gray.type());

    // Start time for latency calculation of CPU function

    struct timespec begin_hw, end_hw, begin_cpu, end_cpu;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    // OpenCV reference function
    cv::GaussianBlur(in_gray, dst, cv::Size(FILTER_WIDTH, FILTER_WIDTH), sigma, sigma, BORDER_CONSTANT);
    dst2 = dst.clone();
    dst3 = dst.clone();
    cv::GaussianBlur(dst2, dst4, cv::Size(FILTER_WIDTH, FILTER_WIDTH), sigma, sigma, BORDER_CONSTANT);
    subtract(dst3, dst4, dst_fin);

    // End time for latency calculation of CPU function

    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;

    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

    // OpenCL section:
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    int rows = in_gray.rows;
    int cols = in_gray.cols;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    unsigned fileBufSize;
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_gaussiandifference");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "gaussiandiference", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, sigma));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, rows));
    OCL_CHECK(err, err = kernel.setArg(4, cols));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray.data,        // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_img.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Write the output of kernel:
    cv::imwrite("output_hls.png", out_img);
    std::cout << "Test Passed " << std::endl;

    std::cout.precision(3);
    std::cout << std::fixed;

    std::cout << "Latency for CPU function is: " << hw_time << "ms" << std::endl;

    return 0;
}
