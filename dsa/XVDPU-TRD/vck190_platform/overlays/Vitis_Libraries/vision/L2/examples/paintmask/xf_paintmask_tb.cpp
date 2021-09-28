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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, in_gray, diff, in_mask;

    // Reading in the gray image:
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    int height = in_img.rows;
    int width = in_img.cols;
    // Allocate memory for output images:
    ocv_ref.create(in_img.rows, in_img.cols, in_img.depth());
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());
    in_mask.create(in_img.rows, in_img.cols, CV_8UC1);

    uint64_t q = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            for (int c = 0; c < in_img.channels(); c++) {
                if ((j > 250) && (j < 750)) {
                    in_mask.data[q + c] = 255;
                } else {
                    in_mask.data[q + c] = 0;
                }
            }
            q += in_img.channels();
        }
    }

    std::vector<unsigned char> color(in_img.channels());
    for (int i = 0; i < in_img.channels(); i++) {
        color[i] = 150;
    }

    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = image_in_size_bytes;
    size_t mask_in_size_bytes = in_img.rows * in_img.cols * sizeof(unsigned char);
    size_t vec_in_size_bytes = in_img.channels() * sizeof(unsigned char);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_paintmask");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "paintmask_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inMask(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inColor(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImg(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inMask));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inColor));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImg));
    OCL_CHECK(err, err = kernel.setArg(4, height));
    OCL_CHECK(err, err = kernel.setArg(5, width));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       in_img.data,         // Pointer to the data to copy
                                       nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inMask,      // buffer on the FPGA
                                            CL_TRUE,            // blocking call
                                            0,                  // buffer offset in bytes
                                            mask_in_size_bytes, // Size in bytes
                                            in_mask.data,       // Pointer to the data to copy
                                            nullptr, &event));

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inColor,    // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_in_size_bytes, // Size in bytes
                                            color.data(),      // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImg, // This buffers data will be read
                            CL_TRUE,       // blocking call
                            0,             // offset
                            image_out_size_bytes,
                            out_img.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Reference function:
    unsigned long long int p = 0;
    for (int i = 0; i < ocv_ref.rows; i++) {
        for (int j = 0; j < ocv_ref.cols; j++) {
            for (int c = 0; c < ocv_ref.channels(); c++) {
                if (in_mask.data[p + c] != 0) {
                    ocv_ref.data[p + c] = color[c];
                } else {
                    ocv_ref.data[p + c] = in_img.data[p + c];
                }
            }
            p += in_img.channels();
        }
    }

    // Write output image
    cv::imwrite("hls_out.jpg", out_img);
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff); // Save the difference image for debugging purpose

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

    std::cout << "INFO: Verification results:" << std::endl;
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
