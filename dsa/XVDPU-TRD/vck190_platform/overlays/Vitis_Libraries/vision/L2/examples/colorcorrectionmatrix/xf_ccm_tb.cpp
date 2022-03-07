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
#include "xf_ccm_config.h"
#include <iostream>

using namespace std;

const float bt2020_bt709_arr[3][3] = {
    {1.6605, -0.5876, -0.0728}, {-0.1246, 1.1329, -0.0083}, {-0.0182, -0.1006, 1.1187}};

const float bt2020_bt709_off[3] = {0.0, 0.0, 0.0};

const float bt709_bt2020_arr[3][3] = {{0.627, 0.329, 0.0433}, {0.0691, 0.92, 0.0113}, {0.0164, 0.088, 0.896}};

const float bt709_bt2020_off[3] = {0.0, 0.0, 0.0};

const float rgb_yuv_601_arr[3][3] = {{0.257, 0.504, 0.098}, {-0.148, -0.291, 0.439}, {0.439, -0.368, -0.071}};

const float rgb_yuv_601_off[3] = {0.0625, 0.500, 0.500};

const float rgb_yuv_709_arr[3][3] = {{0.183, 0.614, 0.062}, {-0.101, -0.338, 0.439}, {0.439, -0.399, -0.040}};

const float rgb_yuv_709_off[3] = {0.0625, 0.500, 0.500};

const float rgb_yuv_2020_arr[3][3] = {
    {0.225613, 0.582282, 0.050928}, {-0.119918, -0.309494, 0.429412}, {0.429412, -0.394875, -0.034537}};

const float rgb_yuv_2020_off[3] = {0.062745, 0.500, 0.500};

const float yuv_rgb_601_arr[3][3] = {{1.164, 0.000, 1.596}, {1.164, -0.813, -0.391}, {1.164, 2.018, 0.000}};

const float yuv_rgb_601_off[3] = {-0.87075, 0.52925, -1.08175};

const float yuv_rgb_709_arr[3][3] = {{1.164, 0.000, 1.793}, {1.164, -0.213, -0.534}, {1.164, 2.115, 0.000}};

const float yuv_rgb_709_off[3] = {-0.96925, 0.30075, -1.13025};

const float yuv_rgb_2020_arr[3][3] = {
    {1.164384, 0.000000, 1.717000}, {1.164384, -0.191603, -0.665274}, {1.164384, 2.190671, 0.000000}};

const float yuv_rgb_2020_off[3] = {-0.931559, 0.355379, -1.168395};

const float full_to_16_235_arr[3][3] = {
    {0.856305, 0.000000, 0.000000}, {0.000000, 0.856305, 0.000000}, {0.000000, 0.000000, 0.856305}};

const float full_to_16_235_off[3] = {0.0625, 0.0625, 0.0625};

const float full_from_16_235_arr[3][3] = {
    {1.167808, 0.000000, 0.000000}, {0.000000, 1.167808, 0.000000}, {0.000000, 0.000000, 1.167808}};

const float full_from_16_235_off[3] = {-0.0729880, -0.0729880, -0.0729880};

void colorcorrectionmatrix(cv::Mat& _src, cv::Mat& _dst) {
    float ccm_matrix[3][3];
    float offsetarray[3];

    switch (XF_CCM_TYPE) {
        case 0:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = bt2020_bt709_arr[i][j];
                }
                offsetarray[i] = bt2020_bt709_off[i];
            }

            break;
        case 1:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = bt709_bt2020_arr[i][j];
                }
                offsetarray[i] = bt709_bt2020_off[i];
            }

            break;
        case 2:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = rgb_yuv_601_arr[i][j];
                }
                offsetarray[i] = rgb_yuv_601_off[i];
            }

            break;
        case 3:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = rgb_yuv_709_arr[i][j];
                }
                offsetarray[i] = rgb_yuv_709_off[i];
            }

            break;
        case 4:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = rgb_yuv_2020_arr[i][j];
                }
                offsetarray[i] = rgb_yuv_2020_off[i];
            }

            break;
        case 5:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = yuv_rgb_601_arr[i][j];
                }
                offsetarray[i] = yuv_rgb_601_off[i];
            }

            break;
        case 6:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = yuv_rgb_709_arr[i][j];
                }
                offsetarray[i] = yuv_rgb_709_off[i];
            }

            break;
        case 7:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = yuv_rgb_2020_arr[i][j];
                }
                offsetarray[i] = yuv_rgb_2020_off[i];
            }

            break;
        case 8:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = full_to_16_235_arr[i][j];
                }
                offsetarray[i] = full_to_16_235_off[i];
            }

            break;
        case 9:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = full_from_16_235_arr[i][j];
                }
                offsetarray[i] = full_from_16_235_off[i];
            }

            break;
        default:
            break;
    }

    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
#if T_8U

            float _src_r = _src.at<cv::Vec3b>(i, j)[0];
            float _src_g = _src.at<cv::Vec3b>(i, j)[1];
            float _src_b = _src.at<cv::Vec3b>(i, j)[2];
#else
            float _src_r = _src.at<cv::Vec3w>(i, j)[0];
            float _src_g = _src.at<cv::Vec3w>(i, j)[1];
            float _src_b = _src.at<cv::Vec3w>(i, j)[2];

#endif
            float value1 = (_src_r * ccm_matrix[0][0]);
            float value2 = (_src_g * ccm_matrix[0][1]);
            float value3 = (_src_b * ccm_matrix[0][2]);

            float value4 = (_src_r * ccm_matrix[1][0]);
            float value5 = (_src_g * ccm_matrix[1][1]);
            float value6 = (_src_b * ccm_matrix[1][2]);

            float value7 = (_src_r * ccm_matrix[2][0]);
            float value8 = (_src_g * ccm_matrix[2][1]);
            float value9 = (_src_b * ccm_matrix[2][2]);

            int value_r = (int)(value1 + value2 + value3 + offsetarray[0]);
            int value_g = (int)(value4 + value5 + value6 + offsetarray[1]);
            int value_b = (int)(value7 + value8 + value9 + offsetarray[2]);
// cout<<value_r<<" "<<value_g<<" "<<value_b<<endl;
#if T_8U
            _dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<unsigned char>(value_r); // value_r;
            _dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<unsigned char>(value_g); // value_g;
            _dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<unsigned char>(value_b); // value_b;
#else
            _dst.at<cv::Vec3w>(i, j)[0] = cv::saturate_cast<unsigned short>(value_r); // value_r;
            _dst.at<cv::Vec3w>(i, j)[1] = cv::saturate_cast<unsigned short>(value_g); // value_g;
            _dst.at<cv::Vec3w>(i, j)[2] = cv::saturate_cast<unsigned short>(value_b); // value_b;
#endif
        }
    }
}

int main(int argc, char** argv) {
    cv::Mat in_img, out_img, out_img_hls, diff;

#if T_8U
    in_img = cv::imread(argv[1], 1); // read image
#else
    in_img = cv::imread(argv[1], -1);                                                 // read image
#endif
    if (!in_img.data) {
        return -1;
    }

    imwrite("in_img.png", in_img);

#if T_8U
    out_img.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img_hls.create(in_img.rows, in_img.cols, CV_8UC3);
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;
#else
    out_img.create(in_img.rows, in_img.cols, CV_16UC3);
    out_img_hls.create(in_img.rows, in_img.cols, CV_16UC3);
    diff.create(in_img.rows, in_img.cols, CV_16UC3);
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned short);
    size_t image_out_size_bytes = image_in_size_bytes;
#endif

    colorcorrectionmatrix(in_img, out_img);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_ccm");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "ccm_accel", &err));

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
