/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_lensshading_config.h"
#include <iostream>
#include <math.h>

using namespace std;
// OpenCV reference function:
void LSC_ref(cv::Mat& _src, cv::Mat& _dst) {
    int center_pixel_pos_x = (_src.cols / 2);
    int center_pixel_pos_y = (_src.rows / 2);
    float max_distance = std::sqrt((_src.rows - center_pixel_pos_y) * (_src.rows - center_pixel_pos_y) +
                                   (_src.cols - center_pixel_pos_x) * (_src.cols - center_pixel_pos_x));

    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                float distance = std::sqrt((center_pixel_pos_y - i) * (center_pixel_pos_y - i) +
                                           (center_pixel_pos_x - j) * (center_pixel_pos_x - j)) /
                                 max_distance;

                float gain = (0.01759 * ((distance + 28.37) * (distance + 28.37))) - 13.36;
#if T_8U
                int value = (_src.at<cv::Vec3b>(i, j)[k] * gain);
                if (value > 255) {
                    value = 255;
                }
                _dst.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<unsigned char>(value);
#endif
#if T_16U
                int value = (_src.at<cv::Vec3w>(i, j)[k] * gain);
                if (value > 65535) {
                    value = 65535;
                }
                _dst.at<cv::Vec3w>(i, j)[k] = cv::saturate_cast<unsigned short>(value);

#endif
            }
        }
    }
}

int main(int argc, char** argv) {
    cv::Mat in_img, out_img, out_img_hls, diff;
#if T_8U
    in_img = cv::imread(argv[1], 1);
#else
    in_img = cv::imread(argv[1], -1);
#endif
    if (!in_img.data) {
        return -1;
    }

    imwrite("in_img.png", in_img);

#if T_8U
    out_img.create(in_img.rows, in_img.cols, in_img.type());
    out_img_hls.create(in_img.rows, in_img.cols, in_img.type());
    diff.create(in_img.rows, in_img.cols, in_img.type());
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;
#endif
#if T_16U
    out_img.create(in_img.rows, in_img.cols, in_img.type());
    out_img_hls.create(in_img.rows, in_img.cols, in_img.type());
    diff.create(in_img.rows, in_img.cols, in_img.type());
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned short);
    size_t image_out_size_bytes = image_in_size_bytes;
#endif

    imwrite("out_img1.png", out_img);
    imwrite("out_img_hls1.png", out_img_hls);

    LSC_ref(in_img, out_img);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_lensshading");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "lensshading_accel", &err));

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

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       in_img.data,         // Pointer to the data to copy
                                       nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_img_hls.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Write output image
    cv::imwrite("hls_out.png", out_img_hls);
    cv::imwrite("ocv_out.png", out_img);

    // Compute absolute difference image
    cv::absdiff(out_img_hls, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 0.0f) {
        fprintf(stderr, " ERROR: Test Failed.\n ");
        return 1;
    }
    std::cout << " Test Passed " << std::endl;

    return 0;
}
