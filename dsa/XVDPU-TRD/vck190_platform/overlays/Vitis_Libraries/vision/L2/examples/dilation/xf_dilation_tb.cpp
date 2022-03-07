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
#include "xf_dilation_config.h"

#include "xcl2.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref;
    cv::Mat diff;

// reading in the image
#if GRAY
    in_img = cv::imread(argv[1], 0);
#else
    in_img = cv::imread(argv[1], 1);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

// create memory for output images
#if GRAY
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);
    out_img.create(in_img.rows, in_img.cols, CV_8UC1);
    diff.create(in_img.rows, in_img.cols, CV_8UC1);

#else
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img.create(in_img.rows, in_img.cols, CV_8UC3);
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));
    cv::dilate(in_img, ocv_ref, element, cv::Point(-1, -1), ITERATIONS, cv::BORDER_CONSTANT);
    cv::imwrite("out_ocv.jpg", ocv_ref);
    /////////////////////	End of OpenCV reference	 ////////////////
    ////////////////////	HLS TOP function call	/////////////////

    unsigned char structure_element[FILTER_SIZE * FILTER_SIZE];

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        structure_element[i] = element.data[i];
    }

    /////////////////////////////////////// CL ////////////////////////

    int height = in_img.rows;
    int width = in_img.cols;

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_dilation");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "dilation_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width * CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer kernelFilterToDevice(context, CL_MEM_READ_ONLY,
                                                   (FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char)), NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, imageToDevice));
    OCL_CHECK(err, err = kernel.setArg(1, imageFromDevice));
    OCL_CHECK(err, err = kernel.setArg(2, kernelFilterToDevice));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(imageToDevice,              // buffer on the FPGA
                                            CL_TRUE,                    // blocking call
                                            0,                          // buffer offset in bytes
                                            (height * width * CH_TYPE), // Size in bytes
                                            in_img.data,                // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(kernelFilterToDevice, // buffer on the FPGA
                                            CL_TRUE,              // blocking call
                                            0,                    // buffer offset in bytes
                                            (FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char)), // Size in bytes
                                            structure_element, // Pointer to the data to copy
                                            nullptr, &event));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel, NULL, &event_sp));

    clWaitForEvents(1, (const cl_event*)&event_sp);

    // Profiling
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(imageFromDevice, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            (height * width * CH_TYPE),
                            out_img.data, // Data will be stored here
                            nullptr, &event);
    queue.finish();
    /////////////////////////////////////// end of CL ////////////////////////

    // Write output image
    cv::imwrite("hw_out.jpg", out_img);

    //////////////////  Compute Absolute Difference ////////////////////
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("out_error.jpg", diff);

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 0) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) return 1;

    return 0;
}
