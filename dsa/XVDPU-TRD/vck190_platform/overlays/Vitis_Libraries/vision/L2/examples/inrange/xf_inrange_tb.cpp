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

    cv::Mat in_img, out_img, ocv_ref, in_gray, diff;

// Reading in the image:
#if RGB
    in_img = cv::imread(argv[1], 1);
#else
    in_img = cv::imread(argv[1], 0);
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    // Create memory for outputs:
    ocv_ref.create(in_img.rows, in_img.cols, in_img.depth());
    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    diff.create(in_img.rows, in_img.cols, in_img.depth());

    // Reference function in OpenCV:
    unsigned char lower_thresh = 50;
    unsigned char upper_thresh = 100;
    int height = in_img.rows;
    int width = in_img.cols;

#if RGB
    cv::inRange(in_img, cv::Scalar(lower_thresh, lower_thresh, lower_thresh),
                cv::Scalar(upper_thresh, upper_thresh, upper_thresh), ocv_ref);
#else
    cv::inRange(in_img, cv::Scalar(lower_thresh), cv::Scalar(upper_thresh), ocv_ref);
#endif

    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * in_img.channels() * sizeof(unsigned char);
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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_inrange");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "inrange_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, lower_thresh));
    OCL_CHECK(err, err = kernel.setArg(2, upper_thresh));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));
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

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_img.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Write output image:
    cv::imwrite("hls_out.jpg", out_img);
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image

    // Results verification:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff); // Save the difference image for debugging purpose

    // Find minimum and maximum differences:
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
#if RGB
            cv::Vec3b v = diff.at<cv::Vec3b>(i, j);
            if (v[0] > 1) cnt++;
            if (v[1] > 1) cnt++;
            if (v[2] > 1) cnt++;

            if (minval > v[0]) minval = v[0];
            if (minval > v[1]) minval = v[1];
            if (minval > v[2]) minval = v[2];

            if (maxval < v[0]) maxval = v[0];
            if (maxval < v[1]) maxval = v[1];
            if (maxval < v[2]) maxval = v[2];
#else
            uchar v = diff.at<uchar>(i, j);
            if (v > 1) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
#endif
        }
    }
    float err_per = 100.0 * (float)cnt / (in_img.rows * in_img.cols * in_img.channels());

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
