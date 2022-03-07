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
#include "xf_erosion_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, in_img1, out_img, ocv_ref;
    cv::Mat in_gray, in_gray1, diff;
#if GRAY
    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);
#else
    // Reading in the image:
    in_gray = cv::imread(argv[1], 1);
#endif
    if (in_gray.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
    int height = in_gray.rows;
    int width = in_gray.cols;
#if GRAY
    // Create memory for output images:
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC1);
    out_img.create(in_gray.rows, in_gray.cols, CV_8UC1);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC1);
#else
    // Create memory for output images:
    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC3);
    out_img.create(in_gray.rows, in_gray.cols, CV_8UC3);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC3);
#endif
    // OpenCV reference:
    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));
    cv::erode(in_gray, ocv_ref, element, cv::Point(-1, -1), ITERATIONS, cv::BORDER_CONSTANT);
    cv::imwrite("out_ocv.jpg", ocv_ref);

    // HLS implementation:
    std::vector<unsigned char> shape(FILTER_SIZE * FILTER_SIZE);

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        shape[i] = element.data[i];
    }

    // OpenCL section:
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * in_gray.channels() * sizeof(unsigned char);
    size_t vec_in_size_bytes = FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char);
    size_t image_out_size_bytes = in_gray.rows * in_gray.cols * in_gray.channels() * sizeof(unsigned char);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_erosion");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "erosion", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inShape(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inShape));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray.data,        // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inShape,    // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_in_size_bytes, // Size in bytes
                                            shape.data(),      // Pointer to the data to copy
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

    // Write output image
    cv::imwrite("hls_out.jpg", out_img);

    //  Compute absolute difference:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("out_error.jpg", diff);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray.rows; i++) {
        for (int j = 0; j < in_gray.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 0) cnt++;
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
