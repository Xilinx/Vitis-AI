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
#include "xf_magnitude_config.h"

#include "xcl2.hpp"

////////////    Reference for L1NORM    //////////
int ComputeMagnitude(cv::Mat gradx, cv::Mat grady, cv::Mat& dst) {
    int row, col;
    int16_t gx, gy, tmp_res;
    int16_t tmp1, tmp2;
    int16_t res;
    for (row = 0; row < gradx.rows; row++) {
        for (col = 0; col < gradx.cols; col++) {
            gx = gradx.at<int16_t>(row, col);
            gy = grady.at<int16_t>(row, col);
            tmp1 = abs(gx);
            tmp2 = abs(gy);
            tmp_res = tmp1 + tmp2;
            res = (int16_t)tmp_res;
            dst.at<int16_t>(row, col) = res;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, in_gray, c_grad_x, c_grad_y, c_grad_x1, c_grad_y1, ocv_ref1, ocv_ref2, ocv_res, out_img, diff;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int filter_size = 3;

    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    /*  convert to gray  */

    cv::Sobel(in_img, c_grad_x, CV_16S, 1, 0, filter_size, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(in_img, c_grad_y, CV_16S, 0, 1, filter_size, scale, delta, cv::BORDER_CONSTANT);

    ocv_ref1.create(in_img.rows, in_img.cols, CV_16S);
    out_img.create(in_img.rows, in_img.cols, CV_16S);
    diff.create(in_img.rows, in_img.cols, CV_16S);

    /////////////////////////////////////// CL ////////////////////////

    int height = in_img.rows;
    int width = in_img.cols;

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_magnitude");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    // Create a kernel:
    OCL_CHECK(err, cl::Kernel krnl(program, "magnitude_accel", &err));

    std::vector<cl::Memory> inBufVec, inBufVec1, outBufVec;
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageToDevice1(context, CL_MEM_READ_ONLY, (height * width * 2), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, (height * width * 2), NULL, &err));

    // Set the kernel arguments
    OCL_CHECK(err, err = krnl.setArg(0, imageToDevice));
    OCL_CHECK(err, err = krnl.setArg(1, imageToDevice1));
    OCL_CHECK(err, err = krnl.setArg(2, imageFromDevice));
    OCL_CHECK(err, err = krnl.setArg(3, height));
    OCL_CHECK(err, err = krnl.setArg(4, width));

    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height * width * 2), c_grad_x.data));
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice1, CL_TRUE, 0, (height * width * 2), c_grad_y.data));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("before kernel .... !!!\n");
    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);
    printf("after kernel .... !!!\n");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height * width * 2), out_img.data);
    q.finish();
/////////////////////////////////////// end of CL ////////////////////////
/////////////////    OpenCV reference  /////////////////
#if L1NORM
    ComputeMagnitude(c_grad_x, c_grad_y, ocv_ref1);
#elif L2NORM
    cv::Sobel(in_img, c_grad_x1, CV_32FC1, 1, 0, filter_size, scale, delta, cv::BORDER_CONSTANT);
    Sobel(in_img, c_grad_y1, CV_32FC1, 0, 1, filter_size, scale, delta, cv::BORDER_CONSTANT);
    magnitude(c_grad_x1, c_grad_y1, ocv_ref2);
#endif

#if L1NORM
    imwrite("ref_img.jpg", ocv_ref1); // save the reference image
    absdiff(ocv_ref1, out_img, diff); // Compute absolute difference image
    imwrite("diff_img.jpg", diff);    // Save the difference image for debugging purpose
#elif L2NORM
    ocv_ref2.convertTo(ocv_res, CV_16S); //  convert from 32F type to 16S type for finding the AbsDiff
    imwrite("ref_img.jpg", ocv_res);     // save the reference image
    absdiff(ocv_res, out_img, diff);     // Compute absolute difference image
    imwrite("diff_img.jpg", diff);       // Save the difference image for debugging purpose
#endif

    // Find minimum and maximum differences

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
    std::cout << "\tMinimum error in intensity = " << minval << std::endl;
    std::cout << "\tMaximum error in intensity = " << maxval << std::endl;
    std::cout << "\tPercentage of pixels above error threshold = " << err_per << std::endl;

    if (err_per > 0.0f) {
        return 1;
    }

    return 0;
}
