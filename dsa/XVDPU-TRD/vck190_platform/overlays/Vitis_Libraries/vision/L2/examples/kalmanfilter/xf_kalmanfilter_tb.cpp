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
#include <time.h>

#include "./xf_kalmanfilter_config.h"
#include "common/xf_headers.hpp"
#include "xcl2.hpp"

void error_check(
    cv::KalmanFilter kf, float* Xout_ptr, float* Uout_ptr, float* Dout_ptr, bool tu_or_mu, float* error_out) {
    uint32_t nan_xf = 0x7fc00000;

    cv::Mat Uout(KF_N, KF_N, CV_32FC1, Uout_ptr);
    cv::Mat Dout = cv::Mat::zeros(KF_N, KF_N, CV_32FC1);
    for (int i = 0; i < KF_N; i++) Dout.at<float>(i, i) = Dout_ptr[i];

    cv::Mat Pout(KF_N, KF_N, CV_32FC1);
    Pout = ((Uout * Dout) * Uout.t());

    float tu_max_error_P = -10000;
    int cnt = 0;

    for (int i = 0; i < KF_N; i++) {
        for (int j = 0; j < KF_N; j++) {
            float kernel_output = Pout.at<float>(i, j);
            float refernce_output;
            if (tu_or_mu == 0)
                refernce_output = (float)kf.errorCovPre.at<double>(i, j);
            else
                refernce_output = (float)kf.errorCovPost.at<double>(i, j);

            float error = fabs(kernel_output - refernce_output);

            uint32_t error_int = *(int*)(float*)&error;

            if (error > tu_max_error_P || error_int == nan_xf) {
                tu_max_error_P = error;
                // std::cout << "ERROR: Difference in results for Pout at (" << i << ","
                // << j << "): " << error <<
                // std::endl;
                cnt++;
            }
        }
    }

    float tu_max_error_X = -10000;

    for (int i = 0; i < KF_N; i++) {
        float kernel_output = Xout_ptr[i];

        float refernce_output;
        if (tu_or_mu == 0)
            refernce_output = (float)kf.statePre.at<double>(i);
        else
            refernce_output = (float)kf.statePost.at<double>(i);

        float error = fabs(kernel_output - refernce_output);

        uint32_t error_int = *(int*)(float*)&error;

        if (error > tu_max_error_X || error_int == nan_xf) {
            tu_max_error_X = error;
            // std::cout << "ERROR: Difference in results for Xout at " << i << ": "
            // << error << std::endl;
            cnt++;
        }
    }

    //  std::cout << "INFO: Percentage of errors = " << (float)cnt * 100 / ((KF_N
    //  * KF_N) + KF_N) << "%" << std::endl;

    if (tu_max_error_X > tu_max_error_P)
        *error_out = tu_max_error_X;
    else
        *error_out = tu_max_error_P;
}

int main(int argc, char* argv[]) {
    // Vector sizes:
    size_t vec_nn_size_bytes = KF_N * KF_N * sizeof(float);
    size_t vec_mn_size_bytes = KF_M * KF_N * sizeof(float);
    size_t vec_nc_size_bytes = KF_N * KF_C * sizeof(float);
    size_t vec_n_size_bytes = KF_N * sizeof(float);
    size_t vec_m_size_bytes = KF_M * sizeof(float);
    size_t vec_c_size_bytes = KF_C * sizeof(float);

    // Vectors to hold input/output data:
    float* A_ptr = (float*)malloc(KF_N * KF_N * sizeof(float));
    float* B_ptr = (float*)malloc(KF_N * KF_C * sizeof(float));
    float* Q_ptr = (float*)malloc(KF_N * KF_N * sizeof(float));
    float* Uq_ptr = (float*)malloc(KF_N * KF_N * sizeof(float));
    float* Dq_ptr = (float*)malloc(KF_N * sizeof(float));
    float* H_ptr = (float*)malloc(KF_M * KF_N * sizeof(float));
    float* X0_ptr = (float*)malloc(KF_N * sizeof(float));
    float* P0_ptr = (float*)malloc(KF_N * KF_N * sizeof(float));
    float* U0_ptr = (float*)malloc(KF_N * KF_N * sizeof(float));
    float* D0_ptr = (float*)malloc(KF_N * sizeof(float));
    float* R_ptr = (float*)malloc(KF_M * sizeof(float));
    float* u_ptr = (float*)malloc(KF_C * sizeof(float));
    float* y_ptr = (float*)malloc(KF_M * sizeof(float));
    float* Xout_ptr = (float*)malloc(KF_N * sizeof(float));
    float* Uout_ptr = (float*)malloc(KF_N * KF_N * sizeof(float));
    float* Dout_ptr = (float*)malloc(KF_N * sizeof(float));

    std::vector<double> A_ptr_dp(KF_N * KF_N);
    std::vector<double> B_ptr_dp(KF_N * KF_C);
    std::vector<double> Uq_ptr_dp(KF_N * KF_N);
    std::vector<double> Dq_ptr_dp(KF_N);
    std::vector<double> H_ptr_dp(KF_M * KF_N);
    std::vector<double> X0_ptr_dp(KF_N);
    std::vector<double> U0_ptr_dp(KF_N * KF_N);
    std::vector<double> D0_ptr_dp(KF_N);
    std::vector<double> R_ptr_dp(KF_M);
    std::vector<double> u_ptr_dp(KF_C);
    std::vector<double> y_ptr_dp(KF_M);

    // Control flag for Xilinx Kalman Filter:
    unsigned char control_flag = 103;

    // Init A:
    int Acnt = 0;
    for (int i = 0; i < KF_N; i++) {
        for (int j = 0; j < KF_N; j++) {
            double val = ((double)rand() / (double)(RAND_MAX)) * 2.0;
            A_ptr_dp[Acnt] = val;
            A_ptr[Acnt++] = (float)val;
        }
    }

    // Init B:
    int Bcnt = 0;
    for (int i = 0; i < KF_N; i++) {
        for (int j = 0; j < KF_C; j++) {
            double val = ((double)rand() / (double)(RAND_MAX)) * 1.0;
            B_ptr_dp[Bcnt] = val;
            B_ptr[Bcnt++] = (float)val;
        }
    }

    // Init H:
    int Hcnt = 0;
    for (int i = 0; i < KF_M; i++) {
        for (int j = 0; j < KF_N; j++) {
            double val = ((double)rand() / (double)(RAND_MAX)) * 0.001;
            H_ptr_dp[Hcnt] = val;
            H_ptr[Hcnt++] = (float)val;
        }
    }

    // Init X0:
    for (int i = 0; i < KF_N; i++) {
        double val = ((double)rand() / (double)(RAND_MAX)) * 5.0;
        X0_ptr_dp[i] = val;
        X0_ptr[i] = (float)val;
    }

    // Init R:
    for (int i = 0; i < KF_M; i++) {
        double val = ((double)rand() / (double)(RAND_MAX)) * 0.01;
        R_ptr_dp[i] = val;
        R_ptr[i] = (float)val;
    }
    // Init U0:
    for (int i = 0; i < KF_N; i++) {
        for (int jn = (-i), j = 0; j < KF_N; jn++, j++) {
            int index = j + i * KF_N;
            if (jn < 0) {
                U0_ptr_dp[index] = 0;
                U0_ptr[index] = 0;
            } else if (jn == 0) {
                U0_ptr_dp[index] = 1;
                U0_ptr[index] = 1;
            } else {
                double val = ((double)rand() / (double)(RAND_MAX)) * 1.0;
                U0_ptr_dp[index] = val;
                U0_ptr[index] = (float)val;
            }
        }
    }

    // Init D0:
    for (int i = 0; i < KF_N; i++) {
        double val = ((double)rand() / (double)(RAND_MAX)) * 1.0;
        D0_ptr_dp[i] = val;
        D0_ptr[i] = (float)val;
    }

    // Init Uq:
    for (int i = 0; i < KF_N; i++) {
        for (int jn = (-i), j = 0; j < KF_N; jn++, j++) {
            int index = j + i * KF_N;
            if (jn < 0) {
                Uq_ptr_dp[index] = 0;
                Uq_ptr[index] = 0;
            } else if (jn == 0) {
                Uq_ptr_dp[index] = 1;
                Uq_ptr[index] = 1;
            } else {
                double val = ((double)rand() / (double)(RAND_MAX)) * 1.0;
                Uq_ptr_dp[index] = val;
                Uq_ptr[index] = (float)val;
            }
        }
    }

    // Init Dq:
    for (int i = 0; i < KF_N; i++) {
        double val = ((double)rand() / (double)(RAND_MAX)) * 1.0;
        Dq_ptr_dp[i] = val;
        Dq_ptr[i] = (float)val;
    }

    // Initialization of cv::Mat objects:
    std::cout << "INFO: Init cv::Mat objects." << std::endl;

    cv::Mat A(KF_N, KF_N, CV_64FC1, A_ptr_dp.data());
    cv::Mat B(KF_N, KF_C, CV_64FC1, B_ptr_dp.data());

    cv::Mat Uq(KF_N, KF_N, CV_64FC1, Uq_ptr_dp.data());
    cv::Mat Dq = cv::Mat::zeros(KF_N, KF_N, CV_64FC1);
    for (int i = 0; i < KF_N; i++) Dq.at<double>(i, i) = Dq_ptr_dp[i];
    cv::Mat Q(KF_N, KF_N, CV_64FC1);
    Q = Uq * Dq * Uq.t();

    cv::Mat H(KF_M, KF_N, CV_64FC1, H_ptr_dp.data());
    cv::Mat X0(KF_N, 1, CV_64FC1);
    for (int i = 0; i < KF_N; i++) X0.at<double>(i) = X0_ptr_dp[i];

    cv::Mat U0(KF_N, KF_N, CV_64FC1, U0_ptr_dp.data());
    cv::Mat D0 = cv::Mat::zeros(KF_N, KF_N, CV_64FC1);
    for (int i = 0; i < KF_N; i++) D0.at<double>(i, i) = D0_ptr_dp[i];
    cv::Mat P0(KF_N, KF_N, CV_64FC1);
    P0 = U0 * D0 * U0.t();

    cv::Mat R = cv::Mat::zeros(KF_M, KF_M, CV_64FC1);
    for (int i = 0; i < KF_M; i++) R.at<double>(i, i) = R_ptr_dp[i];
    cv::Mat uk(KF_C, 1, CV_64FC1);
    cv::Mat zk(KF_M, 1, CV_64FC1);

    std::cout << "INFO: Kalman Filter Verification:" << std::endl;
    std::cout << "\tNumber of state variables: " << KF_N << std::endl;
    std::cout << "\tNumber of measurements: " << KF_M << std::endl;
    std::cout << "\tNumber of control input: " << KF_C << std::endl;

    // Start time for latency calculation of CPU function
    struct timespec begin_hw, end_hw, begin_cpu, end_cpu;
    clock_gettime(CLOCK_REALTIME, &begin_hw);

    // OpenCv Kalman Filter in Double Precision
    cv::KalmanFilter kf(KF_N, KF_M, KF_C, CV_64F);
    kf.statePost = X0;
    kf.errorCovPost = P0;
    kf.transitionMatrix = A;
    kf.processNoiseCov = Q;
    kf.measurementMatrix = H;
    kf.measurementNoiseCov = R;
    kf.controlMatrix = B;

    // Init control parameter:
    for (int i = 0; i < KF_C; i++) {
        double val = ((double)rand() / (double)(RAND_MAX)) * 10.0;
        u_ptr[i] = (float)val;
        uk.at<double>(i) = val;
    }

    // OpenCv Kalman Filter in Double Precision - predict:
    kf.predict(uk);

    // Init measurement parameter:
    for (int i = 0; i < KF_M; i++) {
        double val = ((double)rand() / (double)(RAND_MAX)) * 5.0;
        y_ptr[i] = (float)val;
        zk.at<double>(i) = val;
    }

    // OpenCv Kalman Filter in Double Precision - correct/update:
    kf.correct(zk);

    // Ending time latency calculation of CPU function
    clock_gettime(CLOCK_REALTIME, &end_hw);
    long seconds, nanoseconds;
    double hw_time;

    seconds = end_hw.tv_sec - begin_hw.tv_sec;
    nanoseconds = end_hw.tv_nsec - begin_hw.tv_nsec;
    hw_time = seconds + nanoseconds * 1e-9;
    hw_time = hw_time * 1e3;

    // xfOpenCV Kalman Filter in Single Precision - OpenCL section:

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
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_kalmanfilter");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "kalmanfilter_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inA(context, CL_MEM_READ_ONLY, vec_nn_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inB(context, CL_MEM_READ_ONLY, vec_nc_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inUq(context, CL_MEM_READ_ONLY, vec_nn_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inDq(context, CL_MEM_READ_ONLY, vec_n_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inH(context, CL_MEM_READ_ONLY, vec_mn_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inX0(context, CL_MEM_READ_ONLY, vec_n_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inU0(context, CL_MEM_READ_ONLY, vec_nn_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inD0(context, CL_MEM_READ_ONLY, vec_n_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inR(context, CL_MEM_READ_ONLY, vec_m_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inu(context, CL_MEM_READ_ONLY, vec_c_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_iny(context, CL_MEM_READ_ONLY, vec_m_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outX(context, CL_MEM_WRITE_ONLY, vec_n_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outU(context, CL_MEM_WRITE_ONLY, vec_nn_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outD(context, CL_MEM_WRITE_ONLY, vec_n_size_bytes, NULL, &err));

    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inA));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inB));
    OCL_CHECK(err, err = kernel.setArg(2, buffer_inUq));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_inDq));
    OCL_CHECK(err, err = kernel.setArg(4, buffer_inH));
    OCL_CHECK(err, err = kernel.setArg(5, buffer_inX0));
    OCL_CHECK(err, err = kernel.setArg(6, buffer_inU0));
    OCL_CHECK(err, err = kernel.setArg(7, buffer_inD0));
    OCL_CHECK(err, err = kernel.setArg(8, buffer_inR));
    OCL_CHECK(err, err = kernel.setArg(9, buffer_inu));
    OCL_CHECK(err, err = kernel.setArg(10, buffer_iny));
    OCL_CHECK(err, err = kernel.setArg(11, control_flag));
    OCL_CHECK(err, err = kernel.setArg(12, buffer_outX));
    OCL_CHECK(err, err = kernel.setArg(13, buffer_outU));
    OCL_CHECK(err, err = kernel.setArg(14, buffer_outD));

    // Initialize the buffers:
    cl::Event event;

    // Copy input data to device global memory
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inA,        // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_nn_size_bytes, // Size in bytes
                                            A_ptr,             // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inB,        // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_nc_size_bytes, // Size in bytes
                                            B_ptr,             // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inUq,       // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_nn_size_bytes, // Size in bytes
                                            Uq_ptr,            // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inDq,      // buffer on the FPGA
                                            CL_TRUE,          // blocking call
                                            0,                // buffer offset in bytes
                                            vec_n_size_bytes, // Size in bytes
                                            Dq_ptr,           // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inH,        // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_mn_size_bytes, // Size in bytes
                                            H_ptr,             // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inX0,      // buffer on the FPGA
                                            CL_TRUE,          // blocking call
                                            0,                // buffer offset in bytes
                                            vec_n_size_bytes, // Size in bytes
                                            X0_ptr,           // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inU0,       // buffer on the FPGA
                                            CL_TRUE,           // blocking call
                                            0,                 // buffer offset in bytes
                                            vec_nn_size_bytes, // Size in bytes
                                            U0_ptr,            // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inD0,      // buffer on the FPGA
                                            CL_TRUE,          // blocking call
                                            0,                // buffer offset in bytes
                                            vec_n_size_bytes, // Size in bytes
                                            D0_ptr,           // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inR,       // buffer on the FPGA
                                            CL_TRUE,          // blocking call
                                            0,                // buffer offset in bytes
                                            vec_m_size_bytes, // Size in bytes
                                            R_ptr,            // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_inu,       // buffer on the FPGA
                                            CL_TRUE,          // blocking call
                                            0,                // buffer offset in bytes
                                            vec_c_size_bytes, // Size in bytes
                                            u_ptr,            // Pointer to the data to copy
                                            nullptr, &event));
    OCL_CHECK(err, queue.enqueueWriteBuffer(buffer_iny,       // buffer on the FPGA
                                            CL_TRUE,          // blocking call
                                            0,                // buffer offset in bytes
                                            vec_m_size_bytes, // Size in bytes
                                            y_ptr,            // Pointer to the data to copy
                                            nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));

    // Copy Result from Device Global Memory to Host Local Memory
    queue.enqueueReadBuffer(buffer_outX, // This buffers data will be read
                            CL_TRUE,     // blocking call
                            0,           // offset
                            vec_n_size_bytes,
                            Xout_ptr, // Data will be stored here
                            nullptr, &event);
    queue.enqueueReadBuffer(buffer_outU, // This buffers data will be read
                            CL_TRUE,     // blocking call
                            0,           // offset
                            vec_nn_size_bytes,
                            Uout_ptr, // Data will be stored here
                            nullptr, &event);
    queue.enqueueReadBuffer(buffer_outD, // This buffers data will be read
                            CL_TRUE,     // blocking call
                            0,           // offset
                            vec_n_size_bytes,
                            Dout_ptr, // Data will be stored here
                            nullptr, &event);
    // Clean up:
    queue.finish();

    // Results verification:
    float error;
    error_check(kf, Xout_ptr, Uout_ptr, Dout_ptr, 1, &error);

    if (error < 0.001f) {
        std::cout << "INFO: Test Pass" << std::endl;
        return 0;
    } else {
        fprintf(stderr, "ERROR: Test Fail.\n ");
        return -1;
    }
    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Latency for CPU function is: " << hw_time << "ms" << std::endl;
}
