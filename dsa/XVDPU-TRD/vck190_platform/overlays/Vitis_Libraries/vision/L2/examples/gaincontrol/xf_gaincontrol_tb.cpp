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
#include "xf_gaincontrol_config.h"

#include <time.h>
void gainControlOCV(cv::Mat input, cv::Mat& output, int code, unsigned short rgain, unsigned short bgain) {
    cv::Mat mat = input.clone();
    int height = mat.size().height;
    int width = mat.size().width;
#if T_8U
    typedef uint8_t realSize;
#else
    typedef uint16_t realSize;
#endif
    typedef unsigned int maxSize;
    maxSize pixel;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // std::cout<<"("<<i<<","<<j<<")\t";
            pixel = (maxSize)mat.at<realSize>(i, j); // extracting each pixel
            // std::cout<<"Initial: "<<pixel<<"\t";
            bool cond1, cond2;
            cond1 = (j % 2 == 0);
            cond2 = (j % 2 != 0);
            if (code == XF_BAYER_RG) {
                if (i % 2 == 0 && cond1)
                    pixel = (maxSize)((pixel * rgain) >> 7);
                else if (i % 2 != 0 && cond2)
                    pixel = (maxSize)((pixel * bgain) >> 7);
            } else if (code == XF_BAYER_GR) {
                if (i % 2 == 0 && cond2)
                    pixel = (maxSize)((pixel * rgain) >> 7);
                else if (i % 2 != 0 && cond1)
                    pixel = (maxSize)((pixel * bgain) >> 7);
            } else if (code == XF_BAYER_BG) {
                if (i % 2 == 0 && cond1)
                    pixel = (maxSize)((pixel * bgain) >> 7);
                else if (i % 2 == 0 && cond2)
                    pixel = (maxSize)((pixel * rgain) >> 7);
            } else if (code == XF_BAYER_GB) {
                if (i % 2 == 0 && cond2)
                    pixel = (maxSize)((pixel * bgain) >> 7);
                else if (i % 2 != 0 && cond1)
                    pixel = (maxSize)((pixel * rgain) >> 7);
            }
            // std::cout<<"Final: "<<pixel<<std::endl;
            mat.at<realSize>(i, j) = cv::saturate_cast<realSize>(pixel); // writing each pixel
        }
    }
    output = mat;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path1> \n");
        return -1;
    }

    cv::Mat in_gray, in_gray1, ocv_ref, out_gray, diff, ocv_ref_in1, ocv_ref_in2, inout_gray1, ocv_ref_gw;
#if T_8U
    in_gray = cv::imread(argv[1], 0); // read image
#else
    in_gray = cv::imread(argv[1], -1); // read image
#endif
    if (in_gray.data == NULL) {
        fprintf(stderr, "Cannot open image %s\n", argv[1]);
        return -1;
    }

    ocv_ref.create(in_gray.rows, in_gray.cols, CV_8UC1);
    ocv_ref_gw.create(in_gray.rows, in_gray.cols, CV_8UC1);
    out_gray.create(in_gray.rows, in_gray.cols, CV_8UC1);
    diff.create(in_gray.rows, in_gray.cols, CV_8UC1);

    int height = in_gray.rows;
    int width = in_gray.cols;
    unsigned short rgain = 154;
    unsigned short bgain = 140;

    // TIMER START CODE
    struct timespec begin_hw, end_hw;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    // OpenCV Reference
    gainControlOCV(in_gray, ocv_ref, BFORMAT, rgain, bgain);

    // TIMER END CODE
    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;
    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

#if T_8U
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;
#else
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned short);
    size_t image_out_size_bytes = image_in_size_bytes;
#endif

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_gaincontrol");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "gaincontrol_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(2, in_gray.rows));
    OCL_CHECK(err, err = kernel.setArg(3, in_gray.cols));
    OCL_CHECK(err, err = kernel.setArg(4, rgain));
    OCL_CHECK(err, err = kernel.setArg(5, bgain));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
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
                            out_gray.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    imwrite("out_hls.jpg", out_gray);
    imwrite("ocv_ref.png", ocv_ref);

    cv::absdiff(ocv_ref, out_gray, diff);
    imwrite("error.png", diff); // Save the difference image for debugging purpose

    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 1) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return -1;
    } else
        std::cout << "Test Passed " << std::endl;

    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is " << hw_time << "ms" << std::endl;

    return 0;
}
