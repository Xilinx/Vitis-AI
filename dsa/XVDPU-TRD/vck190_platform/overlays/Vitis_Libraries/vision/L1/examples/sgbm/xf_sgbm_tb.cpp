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
#include "xf_sgbm_config.h"

void compute_census_transform(cv::Mat img, long int* ct) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            long int c = 0;
            for (int ki = i - WINDOW_SIZE / 2; ki <= i + WINDOW_SIZE / 2; ki++) {
                for (int kj = j - WINDOW_SIZE / 2; kj <= j + WINDOW_SIZE / 2; kj++) {
                    unsigned char ref;
                    if (ki < 0 || ki > img.rows - 1 || kj < 0 || kj > img.cols - 1)
                        ref = 0;
                    else
                        ref = img.at<unsigned char>(ki, kj);
                    if (ki != i || kj != j) {
                        c = c << 1;
                        if (ref < img.at<unsigned char>(i, j)) {
                            c += 1;
                        }
                    }
                }
                ct[i * img.cols + j] = c;
            }
        }
    }
} // end compute_census_transform()

int compute_hamming_distance(long int a, long int b) {
    long int tmp = a ^ b;
    int sum = 0;
    while (tmp > 0) {
        short int c = tmp & 0x1;
        sum += c;
        tmp >>= 1;
    }
    return sum;
} // end compute_hamming_distance()

void compute_init_cost(long int* ct1, long int* ct2, int* accumulatedCost, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int d = 0; d < TOTAL_DISPARITY; d++) {
                if (j - d >= 0) {
                    int dist = compute_hamming_distance(ct1[i * cols + j], ct2[i * cols + j - d]);
                    accumulatedCost[(i * cols + j) * TOTAL_DISPARITY + d] = dist;
                } else {
                    int dist = compute_hamming_distance(ct1[i * cols + j], 0);
                    accumulatedCost[(i * cols + j) * TOTAL_DISPARITY + d] = dist;
                }
            }
        }
    }
} // end compute_cost()

void init_Lr(int* Lr, int* accumulatedCost, int sizeOfCpd) {
    for (int r = 0; r < NUM_DIR; r++) {
        for (int i = 0; i < sizeOfCpd; i++) {
            Lr[r * sizeOfCpd + i] = accumulatedCost[i];
        }
    }
} // end init_Lr()

int find_minLri(int* Lrpr, int d, int ndisparity) {
    int minLri = INT_MAX;
    for (int i = 0; i < d - 1; i++) {
        if (minLri > Lrpr[i]) {
            minLri = Lrpr[i];
        }
    }
    for (int i = d + 2; i < ndisparity; i++) {
        if (minLri > Lrpr[i]) {
            minLri = Lrpr[i];
        }
    }
    return minLri;
} // end find_minLri()

int find_min(int a, int b, int c, int d) {
    int minimum = a;
    if (minimum > b) minimum = b;
    if (minimum > c) minimum = c;
    if (minimum > d) minimum = d;
    return minimum;
} // end find_min()

void cost_computation(int* Lr, int* accumulatedCost, int rows, int cols, int numDir, int ndisparity) {
    // Computing cost along 5 directions only. (i,j-1) (i-1,j-1) (i-1,j) (i-1,j+1)
    // (i,j+1)
    int iDisp, jDisp;
    for (int r = 0; r < numDir; r++) {
        if (r == 0) {
            iDisp = 0;
            jDisp = -1;
        } else if (r == 1) {
            iDisp = -1;
            jDisp = -1;
        } else if (r == 2) {
            iDisp = -1;
            jDisp = 0;
        } else if (r == 3) {
            iDisp = -1;
            jDisp = 1;
        } else if (r == 4) {
            iDisp = 0;
            jDisp = 1;
        }

        // Changed the indices of the loop below to accommodate for number of
        // directions = 5. Need to make it more
        // flexible.
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Compute p-r
                int iNorm = i + iDisp;
                int jNorm = j + jDisp;

                int* Lrpr = Lr + ((r * rows + iNorm) * cols + jNorm) * ndisparity;

                // Find min_k{Lr(p-r,k)}
                // int minLrk = find_minLrk(Lrpr);

                for (int d = 0; d < ndisparity; d++) {
                    int Cpd = accumulatedCost[(i * cols + j) * ndisparity + d];

                    int tmp;
                    if ((((r == 0) || (r == 1)) && (j == 0)) || (((r == 1) || (r == 2) || (r == 3)) && (i == 0)) ||
                        ((r == 3) && (j == cols - 1)))
                        tmp = Cpd;

                    else {
                        // Find min_i{Lr(p-r,i)}
                        int minLri = find_minLri(Lrpr, d, ndisparity);
                        int Lrpdm1, Lrpdp1;
                        if (d == 0)
                            Lrpdm1 = INT_MAX - SMALL_PENALTY;
                        else
                            Lrpdm1 = Lrpr[d - 1];
                        if (d == ndisparity - 1)
                            Lrpdp1 = INT_MAX - SMALL_PENALTY;
                        else
                            Lrpdp1 = Lrpr[d + 1];

                        int v2 = std::min(std::min(std::min(minLri, Lrpdp1), Lrpdm1), Lrpr[d]);
                        int v1 = find_min(Lrpr[d], Lrpdm1 + SMALL_PENALTY, Lrpdp1 + SMALL_PENALTY, v2 + LARGE_PENALTY);

                        tmp = Cpd + v1 - v2;
                    }
                    Lr[((r * rows + i) * cols + j) * ndisparity + d] = tmp;
                }
            }
        }
    }
} // end cost_computation()

void cost_aggregation(int* aggregatedCost, int* Lr, int rows, int cols, int ndir, int ndisparity) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int d = 0; d < ndisparity; d++) {
                int* ptr = aggregatedCost + (i * cols + j) * ndisparity + d;
                ptr[0] = 0;
                for (int r = 0; r < ndir; r++) {
                    ptr[0] += Lr[((r * rows + i) * cols + j) * ndisparity + d];
                }
            }
        }
    }
} // end cost_aggregation()

void compute_disparity(float* disparity, int* aggregatedCost, int rows, int cols, int ndisparity) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int* costPtr = aggregatedCost + (i * cols + j) * ndisparity;
            int minCost = costPtr[0];
            int mind = 0;
            for (int d = 1; d < ndisparity; d++) {
                if (costPtr[d] < minCost) {
                    minCost = costPtr[d];
                    mind = d;
                }
            }
            // Interpolation to be added
            //			int p=0,n=0;
            //			if(mind > 0)
            //				p = costPtr[mind-1];
            //			else
            //				p = costPtr[mind+1]; // if there is no
            // previous disparity match with the next
            // disparity
            //
            //			if(mind < ndisparity-1)
            //				n = costPtr[mind+1];
            //			else
            //				n = costPtr[mind-1]; // if there is no
            // next disparity match with the previous
            // disparity
            //
            //			float k = p + n - 2*minCost + __ABS((int)p -
            //(int)n);
            //			int num = p - n;
            //			float delta = 0;
            //			if (k != 0) delta = (float)num/k;
            //
            //			float out_disp = (mind + delta);
            disparity[i * cols + j] = mind; // out_disp;
        }
    }

} // end compute_disparity()

// changed the condition from ">= 0" to "> 0"
int interpolateDisp(cv::Mat disp) {
    int32_t height_ = disp.rows;
    int32_t width_ = disp.cols;
    // for each row do
    for (int32_t v = 0; v < height_; v++) {
        // init counter
        int32_t count = 0;

        // for each pixel do
        for (int32_t u = 0; u < width_; u++) {
            // if disparity valid
            if (disp.at<short>(v, u) > 0) {
                // at least one pixel requires interpolation
                if (count >= 1) {
                    // first and last value for interpolation
                    int32_t u1 = u - count;
                    int32_t u2 = u - 1;

                    // set pixel to min disparity
                    if (u1 > 0 && u2 < width_ - 1) {
                        short d_ipol = std::min(disp.at<short>(v, u1 - 1), disp.at<short>(v, u2 + 1));
                        for (int32_t u_curr = u1; u_curr <= u2; u_curr++) disp.at<short>(v, u_curr) = d_ipol;
                    }
                }

                // reset counter
                count = 0;

                // otherwise increment counter
            } else {
                count++;
            }
        }

        // extrapolate to the left
        for (int32_t u = 0; u < width_; u++) {
            if (disp.at<short>(v, u) > 0) {
                for (int32_t u2 = 0; u2 < u; u2++) disp.at<short>(v, u2) = disp.at<short>(v, u);
                break;
            }
        }

        // extrapolate to the right
        for (int32_t u = width_ - 1; u >= 0; u--) {
            if (disp.at<short>(v, u) > 0) {
                for (int32_t u2 = u + 1; u2 <= width_ - 1; u2++) disp.at<short>(v, u2) = disp.at<short>(v, u);
                break;
            }
        }
    }

    // for each column do
    for (int32_t u = 0; u < width_; u++) {
        // extrapolate to the top
        for (int32_t v = 0; v < height_; v++) {
            if (disp.at<short>(v, u) > 0) {
                for (int32_t v2 = 0; v2 < v; v2++) disp.at<short>(v2, u) = disp.at<short>(v, u);
                break;
            }
        }

        // extrapolate to the bottom
        for (int32_t v = height_ - 1; v >= 0; v--) {
            if (disp.at<short>(v, u) > 0) {
                for (int32_t v2 = v + 1; v2 <= height_ - 1; v2++) disp.at<short>(v2, u) = disp.at<short>(v, u);
                break;
            }
        }
    }
    return 0;
}

int compute_SGM(cv::Mat img1, cv::Mat img2, float* disparity) {
    std::vector<long int> ct1(img1.rows * img1.cols);
    std::vector<long int> ct2(img1.rows * img1.cols);

    // Compute census transform
    compute_census_transform(img1, ct1.data());
    compute_census_transform(img2, ct2.data());

    // Memory to store cost of size height x width x number of disparities
    std::vector<int> accumulatedCost(img1.rows * img1.cols * TOTAL_DISPARITY);

    // Compute initial cost, C(p,d)
    compute_init_cost(ct1.data(), ct2.data(), accumulatedCost.data(), img1.rows, img1.cols);

    // Create array for L(r,p,d)
    std::vector<int> Lr(NUM_DIR * img1.rows * img1.cols * TOTAL_DISPARITY);

    // Initialize Lr(p,d) to C(p,d)
    init_Lr(Lr.data(), accumulatedCost.data(), img1.rows * img1.cols * TOTAL_DISPARITY);

    // Compute cost along different directions
    cost_computation(Lr.data(), accumulatedCost.data(), img1.rows, img1.cols, NUM_DIR, TOTAL_DISPARITY);

    // Array for aggregated cost
    std::vector<int> aggregatedCost(img1.rows * img1.cols * TOTAL_DISPARITY);

    // Cost aggregation
    cost_aggregation(aggregatedCost.data(), Lr.data(), img1.rows, img1.cols, NUM_DIR, TOTAL_DISPARITY);

    //	std::cout<< "computed aggregated cost.. " << std::endl;

    // Disparity computation
    compute_disparity(disparity, aggregatedCost.data(), img1.rows, img1.cols, TOTAL_DISPARITY);

    return 0;
}

void saveDisparityMap(float* disparity, int rows, int cols, int ndisparity, std::string outputFileName) {
    cv::Mat disparityMap(rows, cols, CV_8U);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            disparityMap.at<unsigned char>(i, j) = (unsigned char)(disparity[i * cols + j] /** factor*/);
        }
    }
    cv::imwrite(outputFileName, disparityMap);

} // end saveDisparityMap

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1> <INPUT IMAGE PATH 2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_imgL, in_imgR, hls_out;

    // Reading in images:
    in_imgL = cv::imread(argv[1], 0);
    in_imgR = cv::imread(argv[2], 0);

    if (in_imgL.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
    if (in_imgR.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[2]);
        return EXIT_FAILURE;
    }

    // Allocate memory for the output of kernel:
    hls_out.create(in_imgL.rows, in_imgL.cols, in_imgL.depth());

    // Parameters initialization:
    unsigned char small_penalty = SMALL_PENALTY;
    unsigned char large_penalty = LARGE_PENALTY;

    int height = in_imgL.rows;
    int width = in_imgL.cols;

    semiglobalbm_accel((ap_uint<INPUT_PTR_WIDTH>*)in_imgL.data, (ap_uint<INPUT_PTR_WIDTH>*)in_imgR.data, small_penalty,
                       large_penalty, (ap_uint<OUTPUT_PTR_WIDTH>*)hls_out.data, height, width);

    // Write down the HLS result:
    cv::imwrite("hls_out.png", hls_out);

    // Reference code:
    // Array to store disparity:
    std::vector<float> disparity(in_imgL.rows * in_imgL.cols);

    // computing the disparity:
    compute_SGM(in_imgL, in_imgR, disparity.data());

    // Write disparity to file:
    saveDisparityMap(disparity.data(), in_imgL.rows, in_imgL.cols, TOTAL_DISPARITY, "disp_map.png");

    // Results verification:
    cv::Mat disp_mat(in_imgL.rows, in_imgL.cols, CV_8UC1);
    for (int r = 0; r < in_imgL.rows; r++) {
        for (int c = 0; c < in_imgL.cols; c++) {
            disp_mat.at<unsigned char>(r, c) = (unsigned char)(disparity[r * in_imgL.cols + c]);
        }
    }

    cv::Mat diff;
    diff.create(in_imgL.rows, in_imgL.cols, CV_8UC1);
    int cnt = 0;
    for (int i = 0; i < in_imgL.rows; i++) {
        for (int j = 0; j < in_imgL.cols; j++) {
            int d_val = hls_out.at<unsigned char>(i, j) - disp_mat.at<unsigned char>(i, j);

            if (d_val > 0) {
                diff.at<unsigned char>(i, j) = 255;
                cnt++;
            } else
                diff.at<unsigned char>(i, j) = 0;
        }
    }

    cv::imwrite("diff.png", diff);
    std::cout << "INFO: Number of pixels with errors = " << cnt << std::endl;

    if (cnt > 0) {
        fprintf(stderr, "ERROR: Test Failed.\n ");
        return EXIT_FAILURE;
    } else {
        std::cout << "INFO: Test Pass" << std::endl;
    }

    return 0;
}
