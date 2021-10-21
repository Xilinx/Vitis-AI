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
#include "xf_integral_image_config.h"

#include "xcl2.hpp"

int main(int argc, char** argv) {
    cv::Mat in_img, in_img1, out_img, ocv_ref, ocv_ref1;
    cv::Mat in_gray, in_gray1, diff;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image>\n");
        return -1;
    }

    // Read input image
    in_img = cv::imread(argv[1], 0);
    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return -1;
    }

    // create memory for output images
    ocv_ref.create(in_img.rows, in_img.cols, CV_32S);
    ocv_ref1.create(in_img.rows, in_img.cols, CV_32S);

    cv::integral(in_img, ocv_ref, -1);

    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            ocv_ref1.at<unsigned int>(i, j) = ocv_ref.at<unsigned int>(i + 1, j + 1);
        }
    }

    imwrite("out_ocv.png", ocv_ref1);

    // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_32S);
    out_img.create(in_img.rows, in_img.cols, CV_32S);

    /////////////////////////////////////// CL ////////////////////////

    int height = in_img.rows;
    int width = in_img.cols;

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_integral_image");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "integral_accel", &err));

    // Initialize the buffers:
    cl::Event event;

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, (height * width), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, (height * width * 4), NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = krnl.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = krnl.setArg(1, buffer_outImage));
    OCL_CHECK(err, err = krnl.setArg(2, height));
    OCL_CHECK(err, err = krnl.setArg(3, width));

    OCL_CHECK(err,
              q.enqueueWriteBuffer(buffer_inImage,   // buffer on the FPGA
                                   CL_TRUE,          // blocking call
                                   0,                // buffer offset in bytes
                                   (height * width), // Size in bytes
                                   in_img.data,      // Pointer to the data to copy
                                   nullptr, &event));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // fprintf(stderr,"before kernel");
    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    // fprintf(stderr,"after kernel");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    q.enqueueReadBuffer(buffer_outImage, CL_TRUE, 0, (height * width * 4), out_img.data);
    q.finish();
    /////////////////////////////////////// end of CL ////////////////////////

    // Write output image
    imwrite("hls_out.jpg", out_img);

    // Compute absolute difference image
    absdiff(ocv_ref1, out_img, diff);

    // Save the difference image
    imwrite("diff.png", diff);

    float err_per;
    // xf::cv::analyzeDiff(diff, 1, err_per);
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            unsigned int v = diff.at<unsigned int>(i, j);

            if (v > 0) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }

    err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
