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
#include "xf_histogram_config.h"
#include "xcl2.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_gray, hist_ocv;

#if GRAY
    // reading in the color image
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }
    // cvtColor(in_img, in_img, CV_BGR2GRAY);
    //////////////////	Opencv Reference  ////////////////////////
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&in_img, 1, 0, cv::Mat(), hist_ocv, 1, &histSize, &histRange, 1, 0);

#else
    // reading in the color image
    in_img = cv::imread(argv[1], 1);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image\n");
        return 0;
    }
    //////////////////	Opencv Reference  ////////////////////////
    cv::Mat b_hist, g_hist, r_hist;
    std::vector<cv::Mat> bgr_planes;
    cv::split(in_img, bgr_planes);
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, 1, 0);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, 1, 0);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, 1, 0);
#endif

#if GRAY
    // Create a memory to hold HLS implementation output:
    // Create a memory to hold HLS implementation output:
    std::vector<uint32_t> histogram(histSize);
    size_t image_out_size_bytes = histSize * sizeof(uint32_t);
    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * sizeof(unsigned char);
#else
    // Create a memory to hold HLS implementation output:
    // Create a memory to hold HLS implementation output:
    std::vector<uint32_t> histogram(histSize * 3);
    size_t image_out_size_bytes = histSize * 3 * sizeof(uint32_t);
    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#endif

    int rows = in_img.rows;
    int cols = in_img.cols;

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
    unsigned fileBufSize;
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_histogram");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "histogram_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(2, rows));
    OCL_CHECK(err, err = kernel.setArg(3, cols));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_img.data,         // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            histogram.data(), // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

#if GRAY
    FILE *fp, *fp1;
    fp = fopen("out_hls.txt", "w");
    fp1 = fopen("out_ocv.txt", "w");
    for (int cnt = 0; cnt < 256; cnt++) {
        fprintf(fp, "%u\n", histogram[cnt]);
        uint32_t val = (uint32_t)hist_ocv.at<float>(cnt);
        if (val != histogram[cnt]) {
            fprintf(stderr, "\nTest Failed\n");
            return 1;
        }
        std::cout << "Test Passed " << std::endl;
        fprintf(fp1, "%u\n", val);
    }
    fclose(fp);
    fclose(fp1);
#else
    FILE* total = fopen("total.txt", "w");
    for (int i = 0; i < 768; i++) {
        fprintf(total, "%d\n", histogram[i]);
    }
    fclose(total);
    FILE *fp, *fp1;
    fp = fopen("out_hls.txt", "w");
    fp1 = fopen("out_ocv.txt", "w");
    for (int cnt = 0; cnt < 256; cnt++) {
        fprintf(fp, "%u	%u	%u\n", histogram[cnt], histogram[cnt + 256], histogram[cnt + 512]);
        uint32_t b_val = (uint32_t)b_hist.at<float>(cnt);
        uint32_t g_val = (uint32_t)g_hist.at<float>(cnt);
        uint32_t r_val = (uint32_t)r_hist.at<float>(cnt);
        if ((b_val != histogram[cnt]) && (g_val != histogram[256 + cnt]) && (r_val != histogram[512 + cnt])) {
            fprintf(stderr, "\nTest Failed\n");
            return 1;
        }
        std::cout << "Test Passed " << std::endl;
        fprintf(fp1, "%u	%u	%u\n", b_val, g_val, r_val);
    }
    fclose(fp);
    fclose(fp1);
#endif

    return 0;
}
