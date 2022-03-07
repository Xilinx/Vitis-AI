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
#include "xcl2.hpp"
#include "xf_aec_config.h"
#include <math.h>

// OpenCV reference function:
void AEC_ref(cv::Mat& _src, cv::Mat& _dst) {
    // Temporary matrices for processing

    cv::Mat mask1, subimg;

    cv::Mat yuvimage, yuvimageop, finalop;

    cv::cvtColor(_src, yuvimage, cv::COLOR_BGR2HSV);

    cv::Mat yuvchannels[3];

    split(yuvimage, yuvchannels);

    cv::equalizeHist(yuvchannels[2], yuvchannels[2]);

    cv::merge(yuvchannels, 3, yuvimageop);

    cv::cvtColor(yuvimageop, _dst, cv::COLOR_HSV2BGR);
}

int main(int argc, char** argv) {
    cv::Mat in_img, out_img_hls, diff, img_rgba, out_img, out_img1, ocv_ref;

    in_img = cv::imread(argv[1], 1);
    if (!in_img.data) {
        return -1;
    }

    imwrite("input_3.jpg", in_img);

    int height = in_img.rows;
    int width = in_img.cols;

    out_img.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img_hls.create(in_img.rows, in_img.cols, CV_8UC3);
    diff.create(in_img.rows, in_img.cols, CV_8UC3);

    AEC_ref(in_img, out_img);

    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_aec");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "aec_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(2, in_img.rows));
    OCL_CHECK(err, err = kernel.setArg(3, in_img.cols));

    // Initialize the buffers:
    cl::Event event;

    for (int i = 0; i < 2; i++) {
        // Initialize the buffers:
        cl::Event event;
        OCL_CHECK(err,
                  queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                           CL_TRUE,             // blocking call
                                           0,                   // buffer offset in bytes
                                           image_in_size_bytes, // Size in bytes
                                           in_img.data,         // Pointer to the data to copy
                                           nullptr, &event));

        // Execute the kernel:
        OCL_CHECK(err, err = queue.enqueueTask(kernel));

        queue.finish();

        // Copy Result from Device Global Memory to Host Local Memory
        queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                                CL_TRUE,         // blocking call
                                0,               // offset
                                image_out_size_bytes,
                                out_img_hls.data, // Data will be stored here
                                nullptr, &event);
    }

    // Clean up:
    queue.finish();

    // Write output image
    cv::imwrite("hls_out.jpg", out_img_hls);
    cv::imwrite("ocv_out.jpg", out_img);

    // Compute absolute difference image
    cv::absdiff(out_img_hls, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 0.0f) {
        return 1;
    }
    return 0;
}
