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
#include "xf_channel_combine_config.h"

int main(int argc, char** argv) {
#if FOUR_INPUT
    if (argc != 5) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr,
                "<Executable Name> <input image1 path> <input image2 path> "
                "<input image3 path> <input image4 path>\n");
        return -1;
    }
#endif
#if THREE_INPUT
    if (argc != 4) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr,
                "<Executable Name> <input image1 path> <input image2 path> "
                "<input image3 path> \n");
        return -1;
    }
#endif
#if TWO_INPUT
    if (argc != 3) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image1 path> <input image2 path> \n");
        return -1;
    }
#endif

    cv::Mat in_gray1, in_gray2;
    cv::Mat in_gray3, in_gray4;
    cv::Mat out_img, ocv_ref;
    cv::Mat diff;

    // Reading in the images:
    in_gray1 = cv::imread(argv[1], 0);
    in_gray2 = cv::imread(argv[2], 0);

#if !TWO_INPUT
    in_gray3 = cv::imread(argv[3], 0);
    if ((in_gray3.data == NULL)) {
        fprintf(stderr, "Cannot open input images \n");
        return 0;
    }
#endif
    // creating memory for diff image
    diff.create(in_gray1.rows, in_gray1.cols, CV_TYPE);

#if FOUR_INPUT

    in_gray4 = cv::imread(argv[4], 0);

    if ((in_gray4.data == NULL)) {
        fprintf(stderr, "Cannot open image 4\n");
        return 0;
    }

#endif

    // image height and width
    int height = in_gray1.rows;
    int width = in_gray1.cols;

    // Allocate memory for the output images:
    out_img.create(in_gray1.rows, in_gray1.cols, CV_TYPE);

    // OpenCL section:
    size_t image_in_size_bytes = in_gray1.rows * in_gray1.cols * in_gray1.channels() * sizeof(unsigned char);
    size_t image_out_size_bytes = out_img.rows * out_img.cols * out_img.channels() * sizeof(unsigned char);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_channelcombine");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "channel_combine", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage1(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inImage2(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
#if !TWO_INPUT
    OCL_CHECK(err, cl::Buffer buffer_inImage3(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
#endif
#if FOUR_INPUT
    OCL_CHECK(err, cl::Buffer buffer_inImage4(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
#endif
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

// Set kernel arguments:
#if FOUR_INPUT
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inImage3));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_inImage4));
    OCL_CHECK(err, err = kernel.setArg(4, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(5, height));
    OCL_CHECK(err, err = kernel.setArg(6, width));
#endif

#if THREE_INPUT
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inImage3));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(4, height));
    OCL_CHECK(err, err = kernel.setArg(5, width));
#endif

#if TWO_INPUT
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));
#endif

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage1,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray1.data,       // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage2,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray2.data,       // Pointer to the data to copy
                                            nullptr, &event));
#if !TWO_INPUT
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage3,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray3.data,       // Pointer to the data to copy
                                            nullptr, &event));
#endif
#if FOUR_INPUT
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage4,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray4.data,       // Pointer to the data to copy
                                            nullptr, &event));
#endif
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

#if !TWO_INPUT
    // Write the kernel output image:
    cv::imwrite("hls_out.jpg", out_img);
#endif

    // OpenCV reference:
    std::vector<cv::Mat> bgr_planes;
    cv::Mat merged;
#if (!TWO_INPUT)
    bgr_planes.push_back(in_gray3);
#endif
    bgr_planes.push_back(in_gray2);
    bgr_planes.push_back(in_gray1);

#if FOUR_INPUT
    bgr_planes.push_back(in_gray4);
#endif

    cv::merge(bgr_planes, merged);

// Results verification:
#if !TWO_INPUT
    cv::imwrite("out_ocv.jpg", merged);
    cv::absdiff(merged, out_img, diff);
    cv::imwrite("diff.jpg", diff);

    // Find minimum and maximum differences:
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            cv::Vec4b v = diff.at<cv::Vec4b>(i, j);
            if (v[0] > 0) cnt++;
            if (v[1] > 0) cnt++;
            if (v[2] > 0) cnt++;
#if FOUR_INPUT
            if (v[3] > 0) cnt++;
#endif
            if (minval > v[0]) minval = v[0];
            if (minval > v[1]) minval = v[1];
            if (minval > v[2]) minval = v[2];
#if FOUR_INPUT
            if (minval > v[3]) minval = v[3];
#endif
            if (maxval < v[0]) maxval = v[0];
            if (maxval < v[1]) maxval = v[1];
            if (maxval < v[2]) maxval = v[2];
#if FOUR_INPUT
            if (maxval < v[3]) maxval = v[3];
#endif
        }
    }

    float err_per = 100.0 * (float)cnt / (out_img.rows * out_img.cols * out_img.channels());

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << "%" << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }
#endif
    return 0;
}
