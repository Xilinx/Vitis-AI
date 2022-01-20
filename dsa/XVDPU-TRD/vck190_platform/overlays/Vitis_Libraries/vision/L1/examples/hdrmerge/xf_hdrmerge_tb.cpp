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
#include <math.h>
#include "xf_hdrmerge_config.h"

int g_value_com(unsigned short& value_in, float& alpha, float& ob) {
    float radiance_out = (value_in - ob) / alpha;

    return radiance_out;
}

double compute_datareliabilityweight(float& C, float& mu_h, float& mu_l, float& r) {
    double wr;

    if (r < mu_l)
        wr = exp(-C * (std::pow((r - mu_l), 2)));
    else if (r < mu_h)
        wr = 1;
    else
        wr = exp(-C * (std::pow((r - mu_h), 2)));

    return wr;
}

void luminence_compute(cv::Mat& hdr_1,
                       cv::Mat& hdr_2,
                       cv::Mat& hdr_3,
                       cv::Mat& y1_img_gau,
                       cv::Mat& y2_img_gau,
                       cv::Mat& y3_img_gau,
                       cv::Mat& y_diff1_img,
                       cv::Mat& y_diff2_img,
                       cv::Mat& y_diff3_img,
                       cv::Mat& y1_img,
                       cv::Mat& y2_img,
                       cv::Mat& y3_img) {
    cv::Mat hdr_1_b, hdr_2_b, hdr_3_b;

    hdr_1_b.create(hdr_1.rows + 2, hdr_1.cols + 2, CV_8UC3);
    hdr_2_b.create(hdr_2.rows + 2, hdr_2.cols + 2, CV_8UC3);
    hdr_3_b.create(hdr_3.rows + 2, hdr_3.cols + 2, CV_8UC3);

    int border = floor(3 / 2);

    cv::copyMakeBorder(hdr_1, hdr_1_b, border, border, border, border, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(hdr_2, hdr_2_b, border, border, border, border, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(hdr_3, hdr_3_b, border, border, border, border, cv::BORDER_REPLICATE);

    for (int i = 1; i < hdr_1_b.rows - 1; i++) {
        for (int j = 1; j < hdr_1_b.cols - 1; j++) {
            int value1 = 0.0625 * (hdr_1_b.at<unsigned short>(i - 1, j - 1) + hdr_1_b.at<unsigned short>(i - 1, j + 1) +
                                   hdr_1_b.at<unsigned short>(i + 1, j - 1) + hdr_1_b.at<unsigned short>(i + 1, j + 1));
            int value2 = 0.125 * (hdr_1_b.at<unsigned short>(i - 1, j) + hdr_1_b.at<unsigned short>(i, j - 1) +
                                  hdr_1_b.at<unsigned short>(i + 1, j) + hdr_1_b.at<unsigned short>(i, j + 1));

            int value3 = 0.0625 * (hdr_2_b.at<unsigned short>(i - 1, j - 1) + hdr_2_b.at<unsigned short>(i - 1, j + 1) +
                                   hdr_2_b.at<unsigned short>(i + 1, j - 1) + hdr_2_b.at<unsigned short>(i + 1, j + 1));
            int value4 = 0.125 * (hdr_2_b.at<unsigned short>(i - 1, j) + hdr_2_b.at<unsigned short>(i, j - 1) +
                                  hdr_2_b.at<unsigned short>(i + 1, j) + hdr_2_b.at<unsigned short>(i, j + 1));

            int value5 = 0.0625 * (hdr_3_b.at<unsigned short>(i - 1, j - 1) + hdr_3_b.at<unsigned short>(i - 1, j + 1) +
                                   hdr_3_b.at<unsigned short>(i + 1, j - 1) + hdr_3_b.at<unsigned short>(i + 1, j + 1));
            int value6 = 0.125 * (hdr_3_b.at<unsigned short>(i - 1, j) + hdr_3_b.at<unsigned short>(i, j - 1) +
                                  hdr_3_b.at<unsigned short>(i + 1, j) + hdr_3_b.at<unsigned short>(i, j + 1));

            y1_img.at<unsigned short>(i - 1, j - 1) = value1 + value2 + 0.25 * hdr_1_b.at<unsigned short>(i, j);
            y2_img.at<unsigned short>(i - 1, j - 1) = value3 + value4 + 0.25 * hdr_2_b.at<unsigned short>(i, j);
            y3_img.at<unsigned short>(i - 1, j - 1) = value5 + value6 + 0.25 * hdr_3_b.at<unsigned short>(i, j);
        }
    }

    cv::Mat filter;

    filter.create(3, 3, CV_32F);

    filter.at<float>(0, 0) = filter.at<float>(0, 2) = filter.at<float>(2, 0) = filter.at<float>(2, 2) = 0.0625;
    filter.at<float>(0, 1) = filter.at<float>(1, 0) = filter.at<float>(1, 2) = filter.at<float>(2, 1) = 0.125;
    filter.at<float>(1, 1) = 0.25;

    cv::Point anchor = cv::Point(-1, -1);

    cv::filter2D(y1_img, y1_img_gau, CV_16U, filter, anchor, 0, cv::BORDER_REPLICATE);
    cv::filter2D(y2_img, y2_img_gau, CV_16U, filter, anchor, 0, cv::BORDER_REPLICATE);
    cv::filter2D(y3_img, y3_img_gau, CV_16U, filter, anchor, 0, cv::BORDER_REPLICATE);

    // luminence difference
    for (int i = 0; i < y1_img.rows; i++) {
        for (int j = 0; j < y1_img.cols; j++) {
            y_diff1_img.at<float>(i, j) =
                (float)(((float)abs(y2_img_gau.at<unsigned short>(i, j) - y1_img_gau.at<unsigned short>(i, j))) /
                        std::max(y1_img_gau.at<unsigned short>(i, j), y2_img_gau.at<unsigned short>(i, j)));

            y_diff2_img.at<float>(i, j) = 0;

            y_diff3_img.at<float>(i, j) =
                (float)(((float)abs(y3_img_gau.at<unsigned short>(i, j) - y2_img_gau.at<unsigned short>(i, j))) /
                        std::max(y3_img_gau.at<unsigned short>(i, j), y2_img_gau.at<unsigned short>(i, j)));
        }
    }
}

void chrominance_r_compute(
    cv::Mat& _src1, cv::Mat& _src2, cv::Mat& _src3, cv::Mat& r_img1, cv::Mat& r_img2, cv::Mat& r_img3) {
#if 1 // Bayerformat assuming as RGGB

    for (int i = 0; i < _src1.rows - 2; i++) {
        for (int j = 0; j < _src1.cols - 2; j++) {
            if ((j % 2) == 0 && (i % 2) == 0) {
                r_img1.at<unsigned short>(i, j) = _src1.at<unsigned short>(i, j);
                r_img2.at<unsigned short>(i, j) = _src2.at<unsigned short>(i, j);
                r_img3.at<unsigned short>(i, j) = _src3.at<unsigned short>(i, j);
            }
            if ((j % 2) == 1 && (i % 2) == 0) {
                r_img1.at<unsigned short>(i, j) =
                    0.25 * (_src1.at<unsigned short>(i, j - 1) + _src1.at<unsigned short>(i, j + 1) +
                            _src1.at<unsigned short>(i + 2, j - 1) + _src1.at<unsigned short>(i + 2, j + 1));
                r_img2.at<unsigned short>(i, j) =
                    0.25 * (_src2.at<unsigned short>(i, j - 1) + _src2.at<unsigned short>(i, j + 1) +
                            _src2.at<unsigned short>(i + 2, j - 1) + _src2.at<unsigned short>(i + 2, j + 1));
                r_img3.at<unsigned short>(i, j) =
                    0.25 * (_src3.at<unsigned short>(i, j - 1) + _src3.at<unsigned short>(i, j + 1) +
                            _src3.at<unsigned short>(i + 2, j - 1) + _src3.at<unsigned short>(i + 2, j + 1));
            }
            if ((j % 2) == 0 && (i % 2) == 1) {
                r_img1.at<unsigned short>(i, j) =
                    0.25 * (_src1.at<unsigned short>(i - 1, j) + _src1.at<unsigned short>(i - 1, j + 2) +
                            _src1.at<unsigned short>(i + 1, j) + _src1.at<unsigned short>(i + 1, j + 2));
                r_img2.at<unsigned short>(i, j) =
                    0.25 * (_src2.at<unsigned short>(i - 1, j) + _src2.at<unsigned short>(i - 1, j + 2) +
                            _src2.at<unsigned short>(i + 1, j) + _src2.at<unsigned short>(i + 1, j + 2));
                r_img3.at<unsigned short>(i, j) =
                    0.25 * (_src3.at<unsigned short>(i - 1, j) + _src3.at<unsigned short>(i - 1, j + 2) +
                            _src3.at<unsigned short>(i + 1, j) + _src3.at<unsigned short>(i + 1, j + 2));
            }
            if ((j % 2) == 1 && (i % 2) == 1) {
                r_img1.at<unsigned short>(i, j) =
                    0.25 * (_src1.at<unsigned short>(i - 1, j - 1) + _src1.at<unsigned short>(i + 1, j - 1) +
                            _src1.at<unsigned short>(i + 1, j + 1) + _src1.at<unsigned short>(i - 1, j + 1));
                r_img2.at<unsigned short>(i, j) =
                    0.25 * (_src2.at<unsigned short>(i - 1, j - 1) + _src2.at<unsigned short>(i + 1, j - 1) +
                            _src2.at<unsigned short>(i + 1, j + 1) + _src2.at<unsigned short>(i - 1, j + 1));
                r_img3.at<unsigned short>(i, j) =
                    0.25 * (_src3.at<unsigned short>(i - 1, j - 1) + _src3.at<unsigned short>(i + 1, j - 1) +
                            _src3.at<unsigned short>(i + 1, j + 1) + _src3.at<unsigned short>(i - 1, j + 1));
            }
        }
    }
#endif
}

void chrominance_b_compute(
    cv::Mat& _src1, cv::Mat& _src2, cv::Mat& _src3, cv::Mat& b_img1, cv::Mat& b_img2, cv::Mat& b_img3) {
#if 1 // Bayerformat assuming as RGGB
    cv::Mat hdr_1_b, hdr_2_b, hdr_3_b;

    hdr_1_b.create(_src1.rows + 2, _src1.cols + 2, CV_8UC3);
    hdr_2_b.create(_src2.rows + 2, _src2.cols + 2, CV_8UC3);
    hdr_3_b.create(_src3.rows + 2, _src3.cols + 2, CV_8UC3);

    int border = floor(3 / 2);

    cv::copyMakeBorder(_src1, hdr_1_b, border, border, border, border, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(_src2, hdr_2_b, border, border, border, border, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(_src3, hdr_3_b, border, border, border, border, cv::BORDER_REPLICATE);

    for (int i = 1; i < hdr_1_b.rows - 1; i++) {
        for (int j = 1; j < hdr_1_b.cols - 1; j++) {
            if (((j % 2) == 1 && (i % 2) == 1) || ((j % 2) == 1 && (i % 2) == 0)) {
                b_img1.at<unsigned short>(i - 1, j - 1) =
                    0.25 * (hdr_1_b.at<unsigned short>(i - 1, j - 1) + hdr_1_b.at<unsigned short>(i - 1, j + 1) +
                            hdr_1_b.at<unsigned short>(i + 1, j - 1) + hdr_1_b.at<unsigned short>(i + 1, j + 1));
                b_img2.at<unsigned short>(i - 1, j - 1) =
                    0.25 * (hdr_2_b.at<unsigned short>(i - 1, j - 1) + hdr_2_b.at<unsigned short>(i - 1, j + 1) +
                            hdr_2_b.at<unsigned short>(i + 1, j - 1) + hdr_2_b.at<unsigned short>(i + 1, j + 1));
                b_img3.at<unsigned short>(i - 1, j - 1) =
                    0.25 * (hdr_3_b.at<unsigned short>(i - 1, j - 1) + hdr_3_b.at<unsigned short>(i - 1, j + 1) +
                            hdr_3_b.at<unsigned short>(i + 1, j - 1) + hdr_3_b.at<unsigned short>(i + 1, j + 1));
            }
            if ((j % 2) == 0 && (i % 2) == 1) {
                b_img1.at<unsigned short>(i - 1, j - 1) =
                    0.25 * (hdr_1_b.at<unsigned short>(i - 1, j) + hdr_1_b.at<unsigned short>(i - 1, j + 2) +
                            hdr_1_b.at<unsigned short>(i + 1, j) + hdr_1_b.at<unsigned short>(i + 1, j + 2));
                b_img2.at<unsigned short>(i - 1, j - 1) =
                    0.25 * (hdr_2_b.at<unsigned short>(i - 1, j) + hdr_2_b.at<unsigned short>(i - 1, j + 2) +
                            hdr_2_b.at<unsigned short>(i + 1, j) + hdr_2_b.at<unsigned short>(i + 1, j + 2));
                b_img3.at<unsigned short>(i - 1, j - 1) =
                    0.25 * (hdr_3_b.at<unsigned short>(i - 1, j) + hdr_3_b.at<unsigned short>(i - 1, j + 2) +
                            hdr_3_b.at<unsigned short>(i + 1, j) + hdr_3_b.at<unsigned short>(i + 1, j + 2));
            }
            if ((j % 2) == 0 && (i % 2) == 0) {
                b_img1.at<unsigned short>(i - 1, j - 1) = hdr_1_b.at<unsigned short>(i, j);
                b_img2.at<unsigned short>(i - 1, j - 1) = hdr_2_b.at<unsigned short>(i, j);
                b_img3.at<unsigned short>(i - 1, j - 1) = hdr_3_b.at<unsigned short>(i, j);
            }
        }
    }

#endif
}

void HDR_merge(cv::Mat& _src1,
               cv::Mat& _src2,
               float& alpha,
               float& optical_black_value,
               float& intersec,
               float& rho,
               float& imax,
               float* t,
               cv::Mat& final_img,
               short wr_ocv[NO_EXPS][W_B_SIZE]) {
    int m = NO_EXPS;

    float gamma_out[NO_EXPS] = {0.0, 0.0};
    for (int i = 0; i < m - 1; i++) {
        gamma_out[i] = (rho * (imax - optical_black_value) - optical_black_value * (imax - rho)) /
                       (t[i] * rho + t[i + 1] * (imax - rho));
    }

    float mu_h[NO_EXPS] = {0.0, 0.0};
    float mu_l[NO_EXPS] = {0.0, 0.0};

    for (int i = 0; i < m - 1; i++) {
        if (i == 0) {
            float value = (rho - optical_black_value) / alpha;
            mu_h[i] = value / t[0];
        } else {
            mu_h[i] = gamma_out[i] - (gamma_out[i - 1] - mu_h[i - 1]);
        }

        mu_l[i + 1] = 2 * gamma_out[i] - mu_h[i];
    }

    float value_max = (imax - optical_black_value) / alpha;
    mu_h[m - 1] = value_max / t[m - 1];

    float c_inters = -(log(intersec) / (std::pow((gamma_out[0] - mu_h[0]), 2)));

    double wr[NO_EXPS][W_B_SIZE];

    FILE* fp = fopen("weights.txt", "w");
    for (int i = 0; i < NO_EXPS; i++) {
        for (int j = 0; j < (W_B_SIZE); j++) {
            float rv = (float)(j / t[i]);
            wr[i][j] = compute_datareliabilityweight(c_inters, mu_h[i], mu_l[i], rv);
            wr_ocv[i][j] = wr[i][j] * 16384;
            fprintf(fp, "%d,", wr_ocv[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    cv::Mat final_w1, final_w2;
#if T_8U
    final_img.create(_src1.rows, _src1.cols, CV_8UC1);
#else
    final_img.create(_src1.rows, _src1.cols, CV_16UC1);
#endif
    final_w1.create(_src1.rows, _src1.cols, CV_32FC1);
    final_w2.create(_src1.rows, _src1.cols, CV_32FC1);

    FILE* fp1 = fopen("imagevals_ocv.txt", "w");

    for (int i = 0; i < _src1.rows; i++) {
        for (int j = 0; j < _src1.cols; j++) {
#if T_8U
            int val1 = _src1.at<unsigned char>(i, j);
            int val2 = _src2.at<unsigned char>(i, j);
#else
            int val1 = _src1.at<unsigned short>(i, j);
            int val2 = _src2.at<unsigned short>(i, j);
#endif
            final_w1.at<float>(i, j) = (float)(wr[0][val1]);
            final_w2.at<float>(i, j) = (float)(wr[1][val2]);

            float val_1 = final_w1.at<float>(i, j) *
                          val1; // (g_value_com(_src1.at<unsigned short>(i,j),alpha,optical_black_value)/t[0]);
            float val_2 = final_w2.at<float>(i, j) *
                          val2; //(g_value_com(_src2.at<unsigned short>(i,j),alpha,optical_black_value)/t[1]);

            float sum_wei = final_w1.at<float>(i, j) + final_w2.at<float>(i, j);

            int final_val = (int)((float)(val_1 + val_2) / (float)sum_wei);

            if (final_val > (W_B_SIZE - 1)) {
                final_val = (W_B_SIZE - 1);
            }
            fprintf(fp1, "%d,", final_val);
#if T_8U
            final_img.at<unsigned char>(i, j) = (unsigned char)final_val;
#else
            final_img.at<unsigned short>(i, j) = (unsigned short)final_val;
#endif
        }
        fprintf(fp1, "\n");
    }
    fclose(fp1);
}

static void Mat2MultiBayerAXIvideo(cv::Mat& img, InVideoStrm_t_e_s& AXI_video_strm, unsigned char InColorFormat) {
    int i, j, k, l;

#if T_8U
    unsigned char cv_pix;
#else
    unsigned short cv_pix;
#endif
    ap_axiu<AXI_WIDTH_IN, 1, 1, 1> axi;
    int depth = XF_DTPIXELDEPTH(IN_TYPE, NPIX);

    for (i = 0; i < img.rows; i++) {
        for (j = 0; j < img.cols; j += NPIX) {
            if ((i == 0) && (j == 0)) {
                axi.user = 1;
            } else {
                axi.user = 0;
            }
            if (j == (img.cols - NPIX)) {
                axi.last = 1;
            } else {
                axi.last = 0;
            }
            axi.data = -1;
            for (l = 0; l < NPIX; l++) {
                if (img.depth() == CV_16U)
                    cv_pix = img.at<unsigned short>(i, j + l);
                else
                    cv_pix = img.at<unsigned char>(i, j + l);
                switch (depth) {
                    case 10:
                        xf::cv::AXISetBitFields(axi, (l)*depth, depth, (unsigned short)cv_pix);
                        break;
                    case 12:
                        xf::cv::AXISetBitFields(axi, (l)*depth, depth, (unsigned short)cv_pix);
                        break;
                    case 16:
                        xf::cv::AXISetBitFields(axi, (l)*depth, depth, (unsigned short)cv_pix);
                        break;
                    case CV_8U:
                        xf::cv::AXISetBitFields(axi, (l)*depth, depth, (unsigned char)cv_pix);
                        break;
                    default:
                        xf::cv::AXISetBitFields(axi, (l)*depth, depth, (unsigned char)cv_pix);
                        break;
                }
            }
            axi.keep = -1;
            AXI_video_strm << axi;
        }
    }
}

/*********************************************************************************
 * Function:    MultiPixelAXIvideo2Mat
 * Parameters:  96bit stream with 4 pixels packed
 * Return:      None
 * Description: extract pixels from stream and write to open CV Image
 **********************************************************************************/
static void MultiPixelAXIvideo2Mat_gray(OutVideoStrm_t_e_s& AXI_video_strm, cv::Mat& img, unsigned char ColorFormat) {
    int i, j, k, l;
    ap_axiu<AXI_WIDTH_OUT, 1, 1, 1> axi;

#if T_8U
    unsigned char cv_pix;
#else
    unsigned short cv_pix;
#endif
    int depth = XF_DTPIXELDEPTH(IN_TYPE, NPIX);
    bool sof = 0;

    for (i = 0; i < img.rows; i++) {
        for (j = 0; j < img.cols / NPIX; j++) { // 4 pixels read per iteration
            AXI_video_strm >> axi;
            if ((i == 0) && (j == 0)) {
                if (axi.user.to_int() == 1) {
                    sof = 1;
                } else {
                    j--;
                }
            }
            if (sof) {
                for (l = 0; l < NPIX; l++) {
                    cv_pix = axi.data(l * depth + depth - 1, l * depth);

#if T_8U
                    img.at<unsigned char>(i, (NPIX * j + l)) = cv_pix;
#else
                    img.at<unsigned short>(i, (NPIX * j + l)) = cv_pix;
#endif
                }
            } // if(sof)
        }
    }
}

int main(int argc, char** argv) {
    cv::Mat hdr_img_1, hdr_img_2;

    cv::Mat hls_out, final_ocv;

    float alpha = 1.0f;
    float optical_black_value = 0.0f;
    float intersec = 0.25f;
    float rho = 512;
    float imax = (W_B_SIZE - 1);
    float t[NO_EXPS] = {1.0f, 0.25f}; //{1.0f,0.25f,0.0625f};

    hdr_img_1 = cv::imread(argv[1], -1);
    hdr_img_2 = cv::imread(argv[2], -1);

#if T_8U
    hls_out.create(hdr_img_1.rows, hdr_img_1.cols, CV_8UC1);
    final_ocv.create(hdr_img_1.rows, hdr_img_1.cols, CV_8UC1);
#endif
#if T_16U || T_10U || T_12U
    hls_out.create(hdr_img_1.rows, hdr_img_1.cols, CV_16UC1);
    final_ocv.create(hdr_img_1.rows, hdr_img_1.cols, CV_16UC1);
#endif

    short wr_ocv[NO_EXPS][W_B_SIZE];

    // HDR_merge(hdr_img_1, hdr_img_2, alpha, optical_black_value, intersec, rho, imax, t, final_ocv, wr_ocv);

    int rows = hdr_img_1.rows;
    int cols = hdr_img_1.cols;

    short wr_hls[NO_EXPS * NPIX * W_B_SIZE];

    // FILE *fp = fopen("exposuredata.txt","w");
    for (int k = 0; k < NPIX; k++) {
        for (int i = 0; i < NO_EXPS; i++) {
            for (int j = 0; j < (W_B_SIZE); j++) {
                int index1 = (i + k * NO_EXPS) * W_B_SIZE;
                int index = index1 + j;
                wr_hls[(i + k * NO_EXPS) * W_B_SIZE + j] = wr_ocv[i][j];
                // fprintf(fp,"%d,",wr_ocv[i][j]);
            }
            // fprintf(fp,"\n");
        }
        // fprintf(fp,"\n");
    }
    // fclose(fp);

    InVideoStrm_t_e_s src_axi1;
    InVideoStrm_t_e_s src_axi2;
    OutVideoStrm_t_e_s dst_axi;

    Mat2MultiBayerAXIvideo(hdr_img_1, src_axi1, 0);
    Mat2MultiBayerAXIvideo(hdr_img_2, src_axi2, 0);

    hdrmerge_accel(src_axi1, src_axi2, dst_axi, rows, cols, wr_hls);

    MultiPixelAXIvideo2Mat_gray(dst_axi, hls_out, 0);

    imwrite("hdrimage.png", hls_out);
    imwrite("ocvhdrimage.png", final_ocv);

    cv::Mat diff;

    cv::absdiff(hls_out, final_ocv, diff);

    // Save the difference image for debugging purpose:
    cv::imwrite("error.png", diff);
    float err_per;
    xf::cv::analyzeDiff(diff, 1, err_per);

    if (err_per > 0.0f) {
        return 1;
    }

    printf("hdr merge done");

    return 0;
}
