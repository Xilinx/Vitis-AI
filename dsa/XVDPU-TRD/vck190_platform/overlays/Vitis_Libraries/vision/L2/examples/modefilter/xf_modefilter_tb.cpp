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
#include "xf_modefilter_config.h"

cv::RNG rng(12345);
void mode_filter_rgb(cv::Mat _src, cv::Mat _dst, int win_sz) {
    int win_sz_sq = win_sz * win_sz;
    int window[win_sz_sq];
    cv::Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::Mat _src_border;

    _src_border.create(_src.rows + win_sz - 1, _src.cols + win_sz - 1, CV_8UC3);

    int border = floor(win_sz / 2);

    cv::copyMakeBorder(_src, _src_border, border, border, border, border, cv::BORDER_REPLICATE, value);

    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < _src.rows; i++) {
            for (int j = 0; j < _src.cols; j++) {
                for (int p = 0; p < win_sz; p++) {
                    for (int q = 0; q < win_sz; q++) {
                        // cout<<p<<" "<<q<<" "<<endl;
                        window[q + p * win_sz] = _src_border.at<cv::Vec3b>(i + p, j + q)[k];
                    }
                }

                int max_count = 0, idx = 0;

                for (int m = 0; m < win_sz_sq; m++) {
                    int count = 1;
                    for (int n = m + 1; n < win_sz_sq - 1; n++) {
                        if (window[m] == window[n]) count++;
                    }
                    if (count > max_count) {
                        max_count = count;
                    }
                }

                for (int m = 0; m < win_sz_sq; m++) {
                    int count = 1;
                    for (int n = m + 1; n < win_sz_sq - 1; n++) {
                        if (window[m] == window[n]) count++;
                    }

                    if (count == max_count) {
                        idx = m;
                    }
                }

                _dst.at<cv::Vec3b>(i, j)[k] = window[idx];
            }
        }
    }
    return;
}
void mode_filter_gray(cv::Mat _src, cv::Mat _dst, int win_sz) {
    int win_sz_sq = win_sz * win_sz;
    int window[win_sz_sq];
    int i_1_index = 0, j_1_index = 0, i_plus_index = 0, j_plus_index = 0;

    cv::Mat _src_border;

    _src_border.create(_src.rows + win_sz - 1, _src.cols + win_sz - 1, CV_8UC1);

    int border = floor(win_sz / 2);

    cv::copyMakeBorder(_src, _src_border, border, border, border, border, cv::BORDER_REPLICATE);

    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            for (int p = 0; p < win_sz; p++) {
                for (int q = 0; q < win_sz; q++) {
                    window[q + p * win_sz] = _src_border.at<uchar>(i + p, j + q);
                }
            }
            int max_count = 0, idx = 0;

            for (int i = 0; i < win_sz_sq; i++) {
                int count = 1;
                for (int j = i + 1; j < win_sz_sq - 1; j++) {
                    if (window[i] == window[j]) count++;
                }
                if (count > max_count) {
                    max_count = count;
                }
            }

            for (int i = 0; i < win_sz_sq; i++) {
                int count = 1;
                for (int j = i + 1; j < win_sz_sq - 1; j++) {
                    if (window[i] == window[j]) count++;
                }

                if (count == max_count) {
                    idx = i;
                }
            }

            _dst.at<uchar>(i, j) = window[idx];
        }
    }

    return;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, diff;

//  Reading in the image:
#if GRAY
    in_img = cv::imread(argv[1], 0); // reading in the gray image
#else
    in_img = cv::imread(argv[1], 1); // reading in the color image
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

// create memory for output image
#if GRAY
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);
    out_img.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC1);
#else
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

#if GRAY
    mode_filter_gray(in_img, ocv_ref, WINDOW_SIZE);
#else
    mode_filter_rgb(in_img, ocv_ref, WINDOW_SIZE);
#endif
// OpenCL section:
#if GRAY
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 1 * sizeof(unsigned char);
#else
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#endif
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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_modefilter");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "modefilter_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    std::cout << "kernel args" << std::endl;
    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, in_img.rows));
    OCL_CHECK(err, err = kernel.setArg(2, in_img.cols));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       in_img.data,         // Pointer to the data to copy
                                       nullptr, &event));

    std::cout << "before enqueue task" << std::endl;

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    std::cout << "after enqueue task" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_img.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    std::cout << "after finish" << std::endl;

    // Write down output images:
    cv::imwrite("hls_out.jpg", out_img); // kernel output
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    std::cout << "after imwrite" << std::endl;

    absdiff(ocv_ref, out_img, diff);
    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 0, err_per);

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
