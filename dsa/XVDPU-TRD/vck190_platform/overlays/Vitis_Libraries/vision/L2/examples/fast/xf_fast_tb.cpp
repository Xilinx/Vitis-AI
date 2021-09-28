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
#include "xf_fast_config.h"

#include <time.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_gray, out_img, out_img_ocv, out_hls;

    // Reading in the image:
    in_gray = cv::imread(argv[1], 0);

    if (!in_gray.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    std::vector<cv::KeyPoint> keypoints;

    // Threshold for fast:
    unsigned char threshold = 20;

    // Output allocation from HLS implementation:
    out_hls.create(in_gray.rows, in_gray.cols, in_gray.depth());

    // OpenCL section:
    size_t image_in_size_bytes = in_gray.rows * in_gray.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;

    int rows = in_gray.rows;
    int cols = in_gray.cols;

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_fast");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "fast_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, threshold));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, rows));
    OCL_CHECK(err, err = kernel.setArg(4, cols));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
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
                            out_hls.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Write the kernel output:
    cv::imwrite("hls_out.jpg", out_hls);

    // TIMER START CODE
    struct timespec begin_hw, end_hw;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    // OpenCV reference function
    cv::FAST(in_gray, keypoints, threshold, NMS);

    // TIMER END CODE
    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;
    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

    std::vector<cv::Point> hls_points;
    std::vector<cv::Point> ocv_points;
    std::vector<cv::Point> common_points;
    std::vector<cv::Point> noncommon_points;

    FILE *fp, *fp1;
    fp = fopen("ocvpoints.txt", "w");
    fp1 = fopen("hlspoints.txt", "w");

    int nsize = keypoints.size();

    printf("ocvpoints:%d=\n", nsize);

    for (int i = 0; i < nsize; i++) {
        int x = keypoints[i].pt.x;
        int y = keypoints[i].pt.y;
        ocv_points.push_back(cv::Point(x, y));
        fprintf(fp, "x = %d, y = %d\n", x, y);
    }
    fclose(fp);

    out_img_ocv = in_gray.clone();

    int ocv_x = 0, ocv_y = 0;

    for (unsigned int cnt1 = 0; cnt1 < keypoints.size(); cnt1++) {
        ocv_x = keypoints[cnt1].pt.x;
        ocv_y = keypoints[cnt1].pt.y;
        cv::circle(out_img_ocv, cv::Point(ocv_x, ocv_y), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::imwrite("output_ocv.png", out_img_ocv);
    out_img = in_gray.clone();

    for (int j = 0; j < out_hls.rows; j++) {
        for (int i = 0; i < out_hls.cols; i++) {
            unsigned char value = out_hls.at<unsigned char>(j, i);
            if (value != 0) {
                short int y, x;
                y = j;
                x = i;

                cv::Point tmp;
                tmp.x = i;
                tmp.y = j;

                hls_points.push_back(tmp);
                if (j > 0) cv::circle(out_img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255, 255), 2, 8, 0);
            }
        }
    }

    int nsize1 = hls_points.size();

    int Nocv = ocv_points.size();
    int Nhls = hls_points.size();

    for (int r = 0; r < nsize1; r++) {
        int a, b;
        a = (int)hls_points[r].x;
        b = (int)hls_points[r].y;
        fprintf(fp1, "x = %d, y = %d\n", a, b);
    }
    fclose(fp1);

    for (int j = 0; j < Nocv; j++) {
        for (int k = 0; k < Nhls; k++) {
            if ((ocv_points[j].x == ((hls_points[k].x))) && (ocv_points[j].y == ((hls_points[k].y)))) {
                common_points.push_back(ocv_points[j]);
            }
        }
    }

    FILE* fpt3;
    fpt3 = fopen("common.txt", "w");

    for (unsigned int p = 0; p < common_points.size(); p++) {
        fprintf(fpt3, "x = %d, y = %d\n", common_points[p].x, common_points[p].y);
    }

    fclose(fpt3);

    cv::imwrite("output_hls.png", out_img);

    // Results verification:
    float persuccess, perloss, pergain;

    int totalocv = ocv_points.size();
    int totalhls = hls_points.size();
    int ncommon = common_points.size();

    persuccess = (((float)ncommon / totalhls) * 100);
    perloss = (((float)(totalocv - ncommon) / totalocv) * 100);
    pergain = (((float)(totalhls - ncommon) / totalhls) * 100);

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tCommon = " << ncommon << std::endl;
    std::cout << "\tSuccess = " << persuccess << std::endl;
    std::cout << "\tLoss = " << perloss << std::endl;
    std::cout << "\tGain = " << pergain << std::endl;

    if (persuccess < 80) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }
    std::cout << "Test Passed" << std::endl;

    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is " << hw_time << "ms" << std::endl;

    return 0;
}
