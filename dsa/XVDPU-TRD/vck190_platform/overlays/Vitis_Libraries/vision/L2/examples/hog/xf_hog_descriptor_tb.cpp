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
#include "xf_hog_descriptor_config.h"
#include "ObjDet_reference.hpp"
#include "opencv2/objdetect.hpp"
#include "xcl2.hpp"

#include <time.h>

using namespace std;
using namespace cv;

// for masking the output value
#define operatorAND 0x0000FFFF

// main function
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat img_raw, img;

    // Load the test image:
    img_raw = cv::imread(argv[1], 1); // load as color image through command line argument

    if (!img_raw.data) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

// Converting the image type based on the configuration
#if GRAY_T
    cvtColor(img_raw, img, cv::COLOR_BGR2GRAY);
#elif RGB_T
    cvtColor(img_raw, img, cv::COLOR_BGR2RGB);
#endif

    // Creating the input pointers
    uint16_t image_height = img.rows;
    uint16_t image_width = img.cols;

    int novcpb_tb = XF_BLOCK_HEIGHT / XF_CELL_HEIGHT;
    int nohcpb_tb = XF_BLOCK_WIDTH / XF_CELL_WIDTH;

    int nobpb_tb = XF_NO_OF_BINS * nohcpb_tb * novcpb_tb;

    int novbpw_tb = (XF_WIN_HEIGHT / XF_CELL_HEIGHT) - 1;
    int nohbpw_tb = (XF_WIN_WIDTH / XF_CELL_WIDTH) - 1;

    int nodpw_tb = nobpb_tb * novbpw_tb * nohbpw_tb;

    int novw_tb = ((image_height - XF_WIN_HEIGHT) / XF_WIN_STRIDE) + 1;
    int nohw_tb = ((image_width - XF_WIN_WIDTH) / XF_WIN_STRIDE) + 1;

    int total_no_of_windows = novw_tb * nohw_tb;

    int novb_tb = ((image_height / XF_CELL_HEIGHT) - 1);
    int nohb_tb = ((image_width / XF_CELL_WIDTH) - 1);

#if REPETITIVE_BLOCKS
    int dim_rb = (total_no_of_windows * nodpw_tb) >> 1;
    int no_of_descs_hw = dim_rb;
#elif NON_REPETITIVE_BLOCKS
    int dim_nrb = (nohb_tb * novb_tb * nobpb_tb) >> 1;
    int no_of_descs_hw = dim_nrb;
    int dim_expand = (dim_nrb << 1);
#endif
    int dim = (total_no_of_windows * nodpw_tb);

#if GRAY_T
    int _planes = 1;
#elif RGB_T
    int _planes = 3;
#endif

    std::vector<uint32_t> outMat(1 * no_of_descs_hw);

    // OpenCL section:
    size_t image_in_size_bytes = image_height * image_width * sizeof(unsigned char) * _planes;
    size_t image_out_size_bytes = 1 * no_of_descs_hw * sizeof(uint32_t);

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_hog");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "hog_descriptor_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));

    // Set kernel arguments:
    int rows = img.rows;
    int cols = img.cols;
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_outImage));
    OCL_CHECK(err, err = kernel.setArg(2, rows));
    OCL_CHECK(err, err = kernel.setArg(3, cols));
    OCL_CHECK(err, err = kernel.setArg(4, no_of_descs_hw));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       img.data,            // Pointer to the data to copy
                                       nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            outMat.data(), // Data will be stored here
                            nullptr, &event);

    // Clean up:
    queue.finish();

#if REPETITIVE_BLOCKS
    std::vector<uint32_t> output(dim_rb);
#elif NON_REPETITIVE_BLOCKS
    std::vector<uint32_t> output(dim_nrb);
#endif

#if NON_REPETITIVE_BLOCKS

    for (int i = 0; i < dim_nrb; i++) {
        output[i] = outMat[i];
    }

#elif REPETITIVE_BLOCKS
    std::vector<uint16_t> output1(dim);

    uint32_t temp;
    int high = 15, low = 0;

    int cnt = 0;
    for (int i = 0; i < (dim_rb); i++) {
        temp = outMat[i];
        high = 15;
        low = 0;
        output1[i * 2] = (uint16_t)(temp & 0x0000FFFF);
        output1[(i * 2) + 1] = (uint16_t)(temp >> 16);
        high += 16;
        low += 16;
        cnt++;
    }
#endif

    std::vector<float> out_fl(dim);

// converting the fixed point data to floating point data for comparison
#if REPETITIVE_BLOCKS
    for (int i = 0; i < dim; i++) {
        out_fl[i] = ((float)output1[i] / (float)65536.0);
    }
#elif NON_REPETITIVE_BLOCKS // Reading in the NRB mode and arranging the data
    // for comparison
    std::vector<float> out_fl_tmp(dim_expand);
    int tmp_cnt = 0;

    for (int i = 0; i < dim_nrb; i++) {
        for (int j = 0; j < 2; j++) {
            out_fl_tmp[((i << 1) + j)] = (float)((output[i] >> (j << 4)) & operatorAND) / (float)65536.0;
        }
    }

    int mul_factor = (nohb_tb * nobpb_tb);

    for (int i = 0; i < novw_tb; i++) {
        for (int j = 0; j < nohw_tb; j++) {
            for (int k = 0; k < novbpw_tb; k++) {
                for (int l = 0; l < nohbpw_tb; l++) {
                    for (int m = 0; m < nobpb_tb; m++) {
                        int index = (i * mul_factor) + (j * nobpb_tb) + (k * mul_factor) + (l * nobpb_tb) + m;
                        out_fl[tmp_cnt] = out_fl_tmp[index];
                        tmp_cnt++;
                    }
                }
            }
        }
    }
#endif

#if __XF_BENCHMARK

    // Creating the input pointers
    vector<float> descriptorsValues_cv;
    vector<cv::Point> locations_cv;

    // Start time for latency calculation of CPU function

    struct timespec begin_hw, end_hw, begin_cpu, end_cpu;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    // Opencv implementation:
    cv::HOGDescriptor hog(cv::Size(XF_WIN_WIDTH, XF_WIN_HEIGHT), cv::Size(XF_BLOCK_WIDTH, XF_BLOCK_HEIGHT),
                          cv::Size(XF_CELL_WIDTH, XF_CELL_HEIGHT), cv::Size(XF_CELL_WIDTH, XF_CELL_HEIGHT),
                          XF_NO_OF_BINS);

    hog.compute(img, descriptorsValues_cv, cv::Size(XF_CELL_WIDTH, XF_CELL_HEIGHT), cv::Size(0, 0), locations_cv);

    // End time for latency calculation of CPU function

    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double cpu_time, hw_time;

    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is: " << hw_time << "ms" << std::endl;

#else

    // Reference HOG implementation:
    AURHOGDescriptor d(cv::Size(XF_WIN_WIDTH, XF_WIN_HEIGHT), cv::Size(XF_BLOCK_WIDTH, XF_BLOCK_HEIGHT),
                       cv::Size(XF_CELL_WIDTH, XF_CELL_HEIGHT), cv::Size(XF_CELL_WIDTH, XF_CELL_HEIGHT), XF_NO_OF_BINS);

    // Creating the input pointers
    vector<float> descriptorsValues;
    vector<cv::Point> locations;

    d.AURcompute(img, descriptorsValues, cv::Size(XF_CELL_WIDTH, XF_CELL_HEIGHT), cv::Size(0, 0), locations);

    std::vector<float> ocv_out_fl(dim);

    // Output of the OCV will be in column major form, for comparison reason we
    // convert that into row major
    cmToRmConv(descriptorsValues, ocv_out_fl.data(), total_no_of_windows);

    // Error analysis:
    float acc_diff = 0, max_diff = 0, max_pos = 0, counter_for_diff = 0;
    float ocv_desc_at_max, hls_desc_at_max;

    for (int i = 0; i < dim; i++) {
        float diff = ocv_out_fl[i] - out_fl[i];
        if (diff < 0) diff = (-diff);

        if (diff > max_diff) {
            max_diff = diff;
            max_pos = i;
            ocv_desc_at_max = ocv_out_fl[i];
            hls_desc_at_max = out_fl[i];
        }

        if (diff > 0.01) {
            counter_for_diff++;
        }

        acc_diff += diff;
    }

    float avg_diff = (float)acc_diff / (float)dim;

    // Printing the descriptor details to the console
    std::cout << "INFO: Results details" << std::endl;
    std::cout << "\tImage dimensions: " << img.cols << "width x " << img.rows << "height" << std::endl;
    std::cout << "\tFound " << descriptorsValues.size() << " descriptor values" << std::endl;

    std::cout << "\tDescriptors having difference more than (0.01f): " << counter_for_diff << std::endl;
    std::cout << "\t\tavg_diff:" << avg_diff << std::endl
              << "\t\tmax_diff: " << max_diff << std::endl
              << "\t\tmax_position: " << max_pos << std::endl;
    std::cout << "\t\tocv_desc_val_at_max_diff:" << ocv_desc_at_max << std::endl
              << "\t\thls_desc_val_at_max_diff:" << hls_desc_at_max << std::endl;
    std::cout << std::endl;

    if (max_diff > 0.1f) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    }
    std::cout << "Test Passed " << std::endl;
#endif

    return 0;
}
