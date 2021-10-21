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
#include "xf_arithm_config.h"

int main(int argc, char** argv) {
#if ARRAY
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n", argv[0]);
        return EXIT_FAILURE;
    }
#else
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

#endif
    cv::Mat in_img1, in_img2, in_gray1, in_gray2, out_img, ocv_ref, diff;

#if GRAY
    // Reading in the image:
    in_gray1 = cv::imread(argv[1], 0);

    if (in_gray1.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
#else
    in_gray1 = cv::imread(argv[1], 1);

    if (in_gray1.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
#endif
#if ARRAY
#if GRAY
    in_gray2 = cv::imread(argv[2], 0);

    if (in_gray2.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }
#else
    in_gray2 = cv::imread(argv[2], 1);

    if (in_gray2.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }

#endif
#endif

    int height = in_gray1.rows;
    int width = in_gray1.cols;

#if GRAY
#if T_16S
    /*  convert to 16S type  */
    in_gray1.convertTo(in_gray1, CV_16SC1);
    in_gray2.convertTo(in_gray2, CV_16SC1);
    out_img.create(in_gray1.rows, in_gray1.cols, CV_16SC1);
    ocv_ref.create(in_gray1.rows, in_gray1.cols, CV_16SC1);
    diff.create(in_gray1.rows, in_gray1.cols, CV_16SC1);
#else
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC1);
    ocv_ref.create(in_gray1.rows, in_gray1.cols, CV_8UC1);
    diff.create(in_gray1.rows, in_gray1.cols, CV_8UC1);
#endif
#else
#if T_16S
    /*  convert to 16S type  */
    in_gray1.convertTo(in_gray1, CV_16SC3);
    in_gray2.convertTo(in_gray2, CV_16SC3);
    out_img.create(in_gray1.rows, in_gray1.cols, CV_16SC3);
    ocv_ref.create(in_gray1.rows, in_gray1.cols, CV_16SC3);
    diff.create(in_gray1.rows, in_gray1.cols, CV_16SC3);
#else
    out_img.create(in_gray1.rows, in_gray1.cols, CV_8UC3);
    ocv_ref.create(in_gray1.rows, in_gray1.cols, CV_8UC3);
    diff.create(in_gray1.rows, in_gray1.cols, CV_8UC3);
#endif
#endif

#ifdef FUNCT_MULTIPLY
    float scale = 0.05;
#endif

// OpenCL section:
#if T_16S
    size_t image_in_size_bytes = in_gray1.rows * in_gray1.cols * in_gray1.channels() * sizeof(short int);
    size_t image_out_size_bytes = in_gray1.rows * in_gray1.cols * in_gray1.channels() * sizeof(short int);
#else
    size_t image_in_size_bytes = in_gray1.rows * in_gray1.cols * in_gray1.channels() * sizeof(unsigned char);
    size_t image_out_size_bytes = in_gray1.rows * in_gray1.cols * in_gray1.channels() * sizeof(unsigned char);
#endif

#if SCALAR
    unsigned char scalar[XF_CHANNELS(TYPE, NPC1)];

    for (int i = 0; i < in_gray1.channels(); ++i) {
        scalar[i] = 150;
    }

    size_t vec_in_size_bytes = in_gray1.channels() * sizeof(unsigned char);
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
    printf("device found\n");

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_arithm");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    printf("loaded binary found\n");
    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "arithm_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage1(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
#if SCALAR
    OCL_CHECK(err, cl::Buffer buffer_inVec(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));
#else
    OCL_CHECK(err, cl::Buffer buffer_inImage2(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
#endif
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));
    printf("allocated buffer found\n");
    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage1));
#if SCALAR
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inVec));
#else
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inImage2));
#endif
#ifdef FUNCT_MULTIPLY
    OCL_CHECK(err, err = kernel.setArg(2, scale));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(4, height));
    OCL_CHECK(err, err = kernel.setArg(5, width));
#else
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));
#endif
    printf("finished setting args\n");
    // Initilize buffers:
    cl::Event event;

    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage1,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray1.data,       // Pointer to the data to copy
                                            nullptr, &event));

#if SCALAR
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inVec,      // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_in_size_bytes, // Size in bytes
                                            scalar,            // Pointer to the data to copy
                                            nullptr, &event));
#else
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inImage2,     // buffer on the FPGA
                                            CL_TRUE,             // blocking call
                                            0,                   // buffer offset in bytes
                                            image_in_size_bytes, // Size in bytes
                                            in_gray2.data,       // Pointer to the data to copy
                                            nullptr, &event));
#endif
    printf("finished queing\n");
    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel, NULL, &event));
    clWaitForEvents(1, (const cl_event*)&event);

    printf("finished tsk\n");
    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_img.data, // Data will be stored here
                            nullptr, &event);
    printf("finished readbuffer tsk\n");

    // Clean up:
    queue.finish();

    // Write down the kernel result:
    cv::imwrite("hls_out.jpg", out_img);

    printf("cv_referencestarted\n");

/* OpenCV reference function */
#if ARRAY
#if defined(FUNCT_BITWISENOT)
    cv::CV_FUNCT_NAME(in_gray1, ocv_ref);
#elif defined(FUNCT_ZERO)
    ocv_ref = cv::Mat::zeros(in_gray1.rows, in_gray1.cols, in_gray1.depth());
#else
    cv::CV_FUNCT_NAME(in_gray1, in_gray2, ocv_ref
#ifdef FUNCT_MULTIPLY
                      ,
                      scale
#endif
#ifdef FUNCT_COMPARE
                      ,
                      CV_EXTRA_ARG
#endif
                      );
#endif
#endif

#if SCALAR
#if defined(FUNCT_SET)
    ocv_ref.setTo(cv::Scalar(scalar[0]));
#else
#ifdef FUNCT_SUBRS
    cv::CV_FUNCT_NAME(scalar[0], in_gray1, ocv_ref);
#else
    cv::CV_FUNCT_NAME(in_gray1, scalar[0], ocv_ref
#ifdef FUNCT_COMPARE
                      ,
                      CV_EXTRA_ARG
#endif
                      );
#endif
#endif
#endif

    // Write down the OpenCV outputs:
    cv::imwrite("ref_img.jpg", ocv_ref);

    /* Results verification */
    // Do the diff and save it:
    cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("diff_img.jpg", diff);

    // Find the percentage of pixels above error threshold:
    double minval = 256, maxval = 0;
    int cnt = 0;
    for (int i = 0; i < in_gray1.rows; i++) {
        for (int j = 0; j < in_gray1.cols; j++) {
            uchar v = diff.at<uchar>(i, j);

            if (v > 2) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }

    float err_per = 100.0 * (float)cnt / (in_gray1.rows * in_gray1.cols);

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
