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
#include "xf_resize_config.h"

#include "xcl2.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage : <executable> <input image> <output image height> <output image width>\n");
        return -1;
    }
    cv::Mat img, result_hls, result_ocv, error;

#if GRAY
    // reading in the color image
    img = cv::imread(argv[1], 0);
#else
    img = cv::imread(argv[1], 1);
#endif

    if (!img.data) {
        return -1;
    }

    cv::imwrite("input.png", img);

    int in_width = img.cols;
    int in_height = img.rows;
    int out_height = atoi(argv[2]);
    int out_width = atoi(argv[3]);

#if GRAY
    result_hls.create(cv::Size(out_width, out_height), CV_8UC1);
    result_ocv.create(cv::Size(out_width, out_height), CV_8UC1);
    error.create(cv::Size(out_width, out_height), CV_8UC1);
#else
    result_hls.create(cv::Size(out_width, out_height), CV_8UC3);
    result_ocv.create(cv::Size(out_width, out_height), CV_8UC3);
    error.create(cv::Size(out_width, out_height), CV_8UC3);
#endif

// OpenCL section:
#if GRAY
    size_t image_in_size_bytes = in_height * in_width * 1 * sizeof(unsigned char);
    size_t image_out_size_bytes = out_height * out_width * 1 * sizeof(unsigned char);
#else
    size_t image_in_size_bytes = in_height * in_width * 3 * sizeof(unsigned char);
    size_t image_out_size_bytes = out_height * out_width * 3 * sizeof(unsigned char);
#endif

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_resize");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "resize_accel", &err));

    // Allocate the buffers:
    std::vector<cl::Memory> inBufVec, outBufVec;
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
    OCL_CHECK(err, err = krnl.setArg(1, imageFromDevice));
    OCL_CHECK(err, err = krnl.setArg(2, in_height));
    OCL_CHECK(err, err = krnl.setArg(3, in_width));
    OCL_CHECK(err, err = krnl.setArg(4, out_height));
    OCL_CHECK(err, err = krnl.setArg(5, out_width));

    /* Copy input vectors to memory */
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice,       // buffer on the FPGA
                                        CL_TRUE,             // blocking call
                                        0,                   // buffer offset in bytes
                                        image_in_size_bytes, // Size in bytes
                                        img.data));          // Pointer to the data to copy

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Execute the kernel:
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event_sp));

    clWaitForEvents(1, (const cl_event*)&event_sp);

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    OCL_CHECK(err, q.enqueueReadBuffer(imageFromDevice, // This buffers data will be read
                                       CL_TRUE,         // blocking call
                                       0,               // offset
                                       image_out_size_bytes,
                                       result_hls.data)); // Data will be stored here

    q.finish();
/////////////////////////////////////// end of CL
//////////////////////////////////////////

/*OpenCV resize function*/
#if INTERPOLATION == 0
    cv::resize(img, result_ocv, cv::Size(out_width, out_height), 0, 0, cv::INTER_NEAREST);
#endif
#if INTERPOLATION == 1
    cv::resize(img, result_ocv, cv::Size(out_width, out_height), 0, 0, cv::INTER_LINEAR);
#endif
#if INTERPOLATION == 2
    cv::resize(img, result_ocv, cv::Size(out_width, out_height), 0, 0, cv::INTER_AREA);
#endif

    cv::absdiff(result_hls, result_ocv, error);
    float err_per;
    xf::cv::analyzeDiff(error, 5, err_per);

    if (err_per > 1.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    }
    std::cout << "Test Passed " << std::endl;

    return 0;
}
