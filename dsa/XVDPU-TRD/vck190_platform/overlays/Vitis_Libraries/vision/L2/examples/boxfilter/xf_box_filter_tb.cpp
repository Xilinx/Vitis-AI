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
#include "xf_box_filter_config.h"

#include "xcl2.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_gray, in_conv_img, out_img, ocv_ref, diff;

    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

/*  convert to specific types  */
#if T_8U
    in_img.convertTo(in_conv_img, CV_8U); // Size conversion
    int in_bytes = 1;
#elif T_16U
    in_img.convertTo(in_conv_img, CV_16U); // Size conversion
    int in_bytes = 2;
#elif T_16S
    in_img.convertTo(in_conv_img, CV_16S); // Size conversion
    int in_bytes = 2;
#endif

    ocv_ref.create(in_img.rows, in_img.cols, in_conv_img.depth()); // create memory for output image
    out_img.create(in_img.rows, in_img.cols, in_conv_img.depth()); // create memory for output image
    diff.create(in_img.rows, in_img.cols, in_conv_img.depth());    // create memory for output image

/////////////////    OpenCV reference  /////////////////
#if FILTER_SIZE_3
    cv::boxFilter(in_conv_img, ocv_ref, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
#elif FILTER_SIZE_5
    cv::boxFilter(in_conv_img, ocv_ref, -1, cv::Size(5, 5), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
#elif FILTER_SIZE_7
    cv::boxFilter(in_conv_img, ocv_ref, -1, cv::Size(7, 7), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
#endif

    /////////////////////////////////////// CL ////////////////////////

    int height = in_img.rows;
    int width = in_img.cols;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_box_filter");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "box_filter_accel");

    printf("before  cl buffer .... !!!\n");

    cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width * in_bytes));
    cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, (height * width * in_bytes));

    printf("after  cl buffer .... !!!\n");

    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, height * width * in_bytes, in_conv_img.data);

    // Set the kernel arguments
    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
    krnl.setArg(2, height);
    krnl.setArg(3, width);

    printf("after kernel args .... !!!\n");

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("before kernel .... !!!\n");
    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);
    printf("after kernel .... !!!\n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, height * width * in_bytes, out_img.data);
    q.finish();
    /////////////////////////////////////// end of CL ////////////////////////

    absdiff(ocv_ref, out_img, diff);
    imwrite("diff_img.jpg", diff); // Save the difference image for debugging purpose

    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            uchar v = diff.at<uchar>(i, j);
            if (v > 1) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }
    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols);
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
