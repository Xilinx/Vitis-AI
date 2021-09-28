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
#include "xf_crop_config.h"

#include "xcl2.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
using namespace std;

int main(int argc, char** argv) {
    struct timespec start_time;
    struct timespec end_time;
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img[NUM_ROI], ocv_ref[NUM_ROI], in_gray[NUM_ROI], diff[NUM_ROI], out_img1, in_gray1, diff1,
        ocv_ref1;

#if GRAY
    /*  reading in the gray image  */
    in_img = cv::imread(argv[1], 0);
#else
    in_img = cv::imread(argv[1], 1);
    cvtColor(in_img, in_img, cv::COLOR_BGR2RGBA);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    //	1ST ROI
    unsigned int x_loc[NUM_ROI];
    unsigned int y_loc[NUM_ROI];
    unsigned int ROI_height[NUM_ROI];
    unsigned int ROI_width[NUM_ROI];

    //	2nd ROI
    x_loc[0] = 0;
    y_loc[0] = 0;
    ROI_height[0] = 480;
    ROI_width[0] = 320;

    x_loc[1] = 0;
    y_loc[1] = 0;
    ROI_height[1] = 100;
    ROI_width[1] = 200;

    x_loc[2] = 64;
    y_loc[2] = 64;
    ROI_height[2] = 300;
    ROI_width[2] = 301;

    for (int i = 0; i < NUM_ROI; i++) {
        out_img[i].create(ROI_height[i], ROI_width[i], in_img.type());
        ocv_ref[i].create(ROI_height[i], ROI_width[i], in_img.type());
        diff[i].create(ROI_height[i], ROI_width[i], in_img.type());
    }

    ////////////////  reference code  ////////////////
    //		#if __SDSCC__
    //
    //		perf_counter hw_ctr;
    //		hw_ctr.start();
    //		#endif
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    cv::Rect ROI(x_loc[0], y_loc[0], ROI_width[0], ROI_height[0]);
    ocv_ref[0] = in_img(ROI);
    cv::Rect ROI1(x_loc[1], y_loc[1], ROI_width[1], ROI_height[1]);
    ocv_ref[1] = in_img(ROI1);
    cv::Rect ROI2(x_loc[2], y_loc[2], ROI_width[2], ROI_height[2]);
    ocv_ref[2] = in_img(ROI2);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    //	float diff_latency = (end_time.tv_nsec - start_time.tv_nsec)/1e9 + end_time.tv_sec - start_time.tv_sec;
    //	printf("\latency: %f ", diff_latency);
    //		#if __SDSCC__
    //		hw_ctr.stop();
    //		uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
    //		#endif

    //////////////////  end opencv reference code//////////

    /////////////////////////////////////// CL ////////////////////////
    int height = in_img.rows;
    int width = in_img.cols;

    int* roi = (int*)malloc(NUM_ROI * 4 * sizeof(int));
    for (int i = 0, j = 0; i < (NUM_ROI * 4); j++, i += 4) {
        roi[i] = x_loc[j];
        roi[i + 1] = y_loc[j];
        roi[i + 2] = ROI_height[j];
        roi[i + 3] = ROI_width[j];
    }

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

    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_crop");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "crop_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (in_img.rows * in_img.cols * INPUT_CH_TYPE),
                                            NULL, &err));
    OCL_CHECK(err, cl::Buffer structToDeviceroi(context, CL_MEM_READ_ONLY, (NUM_ROI * 4 * sizeof(int)), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceroi1(context, CL_MEM_WRITE_ONLY,
                                                  (ROI_height[0] * ROI_width[0] * OUTPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceroi2(context, CL_MEM_WRITE_ONLY,
                                                  (ROI_height[1] * ROI_width[1] * OUTPUT_CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDeviceroi3(context, CL_MEM_WRITE_ONLY,
                                                  (ROI_height[2] * ROI_width[2] * OUTPUT_CH_TYPE), NULL, &err));

    printf("finished buffer creation task\n");

    // Set the kernel arguments
    OCL_CHECK(err, err = kernel.setArg(0, imageToDevice));
    OCL_CHECK(err, err = kernel.setArg(1, imageFromDeviceroi1));
    OCL_CHECK(err, err = kernel.setArg(2, imageFromDeviceroi2));
    OCL_CHECK(err, err = kernel.setArg(3, imageFromDeviceroi3));
    OCL_CHECK(err, err = kernel.setArg(4, structToDeviceroi));
    OCL_CHECK(err, err = kernel.setArg(5, height));
    OCL_CHECK(err, err = kernel.setArg(6, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(imageToDevice,                               // buffer on the FPGA
                                            CL_TRUE,                                     // blocking call
                                            0,                                           // buffer offset in bytes
                                            (in_img.rows * in_img.cols * INPUT_CH_TYPE), // Size in bytes
                                            in_img.data,                                 // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(structToDeviceroi,           // buffer on the FPGA
                                            CL_TRUE,                     // blocking call
                                            0,                           // buffer offset in bytes
                                            (NUM_ROI * 4 * sizeof(int)), // Size in bytes
                                            roi,                         // Pointer to the data to copy
                                            nullptr, &event));

    printf("finished enqueueWriteBuffer task\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;
    printf("started kernel execution\n");

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel, NULL, &event_sp));

    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("finished kernel execution\n");
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(imageFromDeviceroi1, // This buffers data will be read
                            CL_TRUE,             // blocking call
                            0,                   // offset
                            (ROI_height[0] * ROI_width[0] * OUTPUT_CH_TYPE),
                            out_img[0].data, // Data will be stored here
                            nullptr, &event);
    queue.enqueueReadBuffer(imageFromDeviceroi2, // This buffers data will be read
                            CL_TRUE,             // blocking call
                            0,                   // offset
                            (ROI_height[1] * ROI_width[1] * OUTPUT_CH_TYPE),
                            out_img[1].data, // Data will be stored here
                            nullptr, &event);
    queue.enqueueReadBuffer(imageFromDeviceroi1, // This buffers data will be read
                            CL_TRUE,             // blocking call
                            0,                   // offset
                            (ROI_height[2] * ROI_width[2] * OUTPUT_CH_TYPE),
                            out_img[2].data, // Data will be stored here
                            nullptr, &event);

    queue.finish();
    printf("write output buffer\n");
    /////////////////////////////////////// end of CL /////////////////////
    char hls_strg[30];
    char ocv_strg[30];
    char diff_strg[30];
    // Write output image
    for (int i = 0; i < NUM_ROI; i++) {
        sprintf(hls_strg, "out_img[%d].jpg", i);
        sprintf(ocv_strg, "ocv_ref[%d].jpg", i);
        sprintf(diff_strg, "diff_img[%d].jpg", i);
        cv::imwrite(hls_strg, out_img[i]); // hls image
        cv::imwrite(ocv_strg, ocv_ref[i]); // reference image
        cv::absdiff(ocv_ref[i], out_img[i], diff[i]);
        cv::imwrite(diff_strg, diff[i]); // Save the difference image for debugging purpose
    }

    //	 Find minimum and maximum differences.
    for (int roi = 0; roi < NUM_ROI; roi++) {
        double minval = 256, maxval1 = 0;
        int cnt = 0;
        for (int i = 0; i < ocv_ref[0].rows; i++) {
            for (int j = 0; j < ocv_ref[0].cols; j++) {
                uchar v = diff[0].at<uchar>(i, j);
                if (v > 1) cnt++;
                if (minval > v) minval = v;
                if (maxval1 < v) maxval1 = v;
            }
        }
        float err_per = 100.0 * (float)cnt / (ocv_ref[0].rows * ocv_ref[0].cols);
        std::cout << "\tMinimum error in intensity = " << minval << std::endl;
        std::cout << "\tMaximum error in intensity = " << maxval1 << std::endl;
        std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

        if (err_per > 0.0f) {
            return 1;
        }
    }
    return 0;
}
