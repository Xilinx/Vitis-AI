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

#include "./xf_kalmanfilter_config.h"
#include "common/xf_headers.hpp"

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
    int INIT_EN = 1;
    int TIMEUPDATE_EN = 2;
    int MEASUPDATE_EN = 4;
    int XOUT_EN_TU = 8;
    int UDOUT_EN_TU = 16;
    int XOUT_EN_MU = 32;
    int UDOUT_EN_MU = 64;

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

    ap_uint<PTR_WIDTH>* A_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* B_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * KF_C * sizeof(float));
    ap_uint<PTR_WIDTH>* Uq_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* Dq_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* H_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_M * KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* X0_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* U0_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* D0_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* R_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_M * sizeof(float));
    ap_uint<PTR_WIDTH>* u_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_C * sizeof(float));
    ap_uint<PTR_WIDTH>* y_ptr_in = (ap_uint<PTR_WIDTH>*)malloc(KF_M * sizeof(float));
    ap_uint<PTR_WIDTH>* X_ptr_out = (ap_uint<PTR_WIDTH>*)malloc(KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* U_ptr_out = (ap_uint<PTR_WIDTH>*)malloc(KF_N * KF_N * sizeof(float));
    ap_uint<PTR_WIDTH>* D_ptr_out = (ap_uint<PTR_WIDTH>*)malloc(KF_N * sizeof(float));

    A_ptr_in = (ap_uint<PTR_WIDTH>*)A_ptr;
    B_ptr_in = (ap_uint<PTR_WIDTH>*)B_ptr;
    Uq_ptr_in = (ap_uint<PTR_WIDTH>*)Uq_ptr;
    Dq_ptr_in = (ap_uint<PTR_WIDTH>*)Dq_ptr;
    H_ptr_in = (ap_uint<PTR_WIDTH>*)H_ptr;
    X0_ptr_in = (ap_uint<PTR_WIDTH>*)X0_ptr;
    U0_ptr_in = (ap_uint<PTR_WIDTH>*)U0_ptr;
    D0_ptr_in = (ap_uint<PTR_WIDTH>*)D0_ptr;
    R_ptr_in = (ap_uint<PTR_WIDTH>*)R_ptr;
    u_ptr_in = (ap_uint<PTR_WIDTH>*)u_ptr;
    y_ptr_in = (ap_uint<PTR_WIDTH>*)y_ptr;

    // Init + Time Update + Measurement Update: Xilinx Kalman filter in Single
    // Precision
    kalmanfilter_accel(A_ptr_in,
#if KF_C != 0
                       B_ptr_in,
#endif
                       Uq_ptr_in, Dq_ptr_in, H_ptr_in, X0_ptr_in, U0_ptr_in, D0_ptr_in, R_ptr_in,
#if KF_C != 0
                       u_ptr_in,
#endif
                       y_ptr_in, INIT_EN + TIMEUPDATE_EN + MEASUPDATE_EN + XOUT_EN_MU + UDOUT_EN_MU, X_ptr_out,
                       U_ptr_out, D_ptr_out);

    Xout_ptr = (float*)X_ptr_out;
    Uout_ptr = (float*)U_ptr_out;
    Dout_ptr = (float*)D_ptr_out;

    // Results verification:
    float error;
    error_check(kf, Xout_ptr, Uout_ptr, Dout_ptr, 1, &error);

    if (error < 0.001f) {
        std::cout << "INFO: Test Pass" << std::endl;
        return 0;
    } else {
        fprintf(stderr, "ERROR: Test Fail\n ");

        return -1;
    }
}
