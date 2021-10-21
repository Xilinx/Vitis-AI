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
#include "xf_reduce_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, dst_hls, ocv_ref, in_gray, diff, in_mask;

    // Reading in the image:
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

#if DIM
    if ((REDUCTION_OP == cv::REDUCE_AVG) || (REDUCTION_OP == cv::REDUCE_SUM)) {
        dst_hls.create(in_img.rows, 1, CV_32SC1);
        ocv_ref.create(in_img.rows, 1, CV_32SC1);

    } else {
        dst_hls.create(in_img.rows, 1, CV_8UC1);
        ocv_ref.create(in_img.rows, 1, CV_8UC1);
    }
#else
    if ((REDUCTION_OP == cv::REDUCE_AVG) || (REDUCTION_OP == cv::REDUCE_SUM)) {
        dst_hls.create(1, in_img.cols, CV_32SC1);
        ocv_ref.create(1, in_img.cols, CV_32SC1);
    } else {
        dst_hls.create(1, in_img.cols, CV_8UC1);
        ocv_ref.create(1, in_img.cols, CV_8UC1);
    }
#endif

    unsigned char dimension = DIM;
    size_t image_out_size_bytes;
    // OpenCL section:
    size_t image_in_size_bytes = in_img.rows * in_img.cols * sizeof(unsigned char);
    if ((REDUCTION_OP == XF_REDUCE_AVG) || (REDUCTION_OP == XF_REDUCE_SUM)) {
        image_out_size_bytes = dst_hls.rows * dst_hls.cols * sizeof(int);
    } else {
        image_out_size_bytes = dst_hls.rows * dst_hls.cols * sizeof(unsigned char);
    }

    int height = in_img.rows;
    int width = in_img.cols;

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_reduce");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "reduce_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, dimension));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));

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
    OCL_CHECK(err, err = queue.enqueueTask(kernel, NULL, &event));
    clWaitForEvents(1, (const cl_event*)&event);

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            (ap_uint<PTR_OUT_WIDTH>*)dst_hls.data, // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

    // Reference function
    if ((CV_REDUCE == cv::REDUCE_AVG) || (CV_REDUCE == cv::REDUCE_SUM))
        cv::reduce(in_img, ocv_ref, DIM, CV_REDUCE, CV_32SC1); // avg, sum
    else
        cv::reduce(in_img, ocv_ref, DIM, CV_REDUCE, CV_8UC1);

    // Results verification:
    FILE* fp = fopen("hls", "w");
    FILE* fp1 = fopen("cv", "w");
    int err_cnt = 0;

#if DIM == 1
    for (unsigned int i = 0; i < dst_hls.rows; i++) {
        fprintf(fp, "%d\n", (unsigned char)dst_hls.data[i]);
        fprintf(fp1, "%d\n", ocv_ref.data[i]);
        unsigned int diff = ocv_ref.data[i] - (unsigned char)dst_hls.data[i];
        if (diff > 1) err_cnt++;
    }

    std::cout << "INFO: Percentage of pixels with an error = " << (float)err_cnt * 100 / (float)dst_hls.rows << "%"
              << std::endl;

#endif
#if DIM == 0
    for (int i = 0; i < dst_hls.cols; i++) {
        fprintf(fp, "%d\n", (unsigned char)dst_hls.data[i]);
        fprintf(fp1, "%d\n", ocv_ref.data[i]);
        unsigned int diff = ocv_ref.data[i] - (unsigned char)dst_hls.data[i];
        if (diff > 1) err_cnt++;
    }

    std::cout << "INFO: Percentage of pixels with an error = " << (float)err_cnt * 100 / (float)dst_hls.cols << "%"
              << std::endl;

#endif
    fclose(fp);
    fclose(fp1);
    printf("after file write\n");
    if (err_cnt > 0) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }

    return 0;
}
