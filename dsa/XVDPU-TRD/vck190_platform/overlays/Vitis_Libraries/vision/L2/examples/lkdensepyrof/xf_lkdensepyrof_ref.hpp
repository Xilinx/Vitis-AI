/*
 * Copyright 2021 Xilinx, Inc.
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

#include <stdio.h>
#include <fstream>
#include <iostream>

#define DEBUG 0

#define EPSILON 0.003
#define MEDIAN_FILTER_SIZE 5
#define minEigThreshold 0.0001

//#define MAX(X,Y) (X>Y?(X):(Y))

using namespace std;

void write_to_file_float(cv::Mat img, char* filename);
void write_to_file_short(cv::Mat img, char* filename);
void write_result_to_image(cv::Mat imgNext, cv::Mat glx, cv::Mat gly, int level);

void find_ix_iy(cv::Mat img_c, cv::Mat& gx, cv::Mat& gy) {
    int xp1, xm1, yp1, ym1;
    for (int i = 0; i < img_c.rows; i++) {
        if (i == 0) {
            yp1 = i + 1;
            ym1 = 0;
        } else if (i == img_c.rows - 1) {
            yp1 = i;
            ym1 = i - 1;
        } else {
            yp1 = i + 1;
            ym1 = i - 1;
        }
        for (int j = 0; j < img_c.cols; j++) {
            if (j == 0) {
                xp1 = j + 1;
                xm1 = 0;
            } else if (j == img_c.cols - 1) {
                xp1 = j;
                xm1 = j - 1;
            } else {
                xp1 = j + 1;
                xm1 = j - 1;
            }
            gx.at<short int>(i, j) = (short int)(img_c.at<unsigned char>(i, xp1) - img_c.at<unsigned char>(i, xm1));
            gy.at<short int>(i, j) = (short int)(img_c.at<unsigned char>(yp1, j) - img_c.at<unsigned char>(ym1, j));
        }
    }
}
void constructGradientMatrix(cv::Mat gx, cv::Mat gy, int winsize, float* matrixG) {
#if DEBUG
    FILE* fpix2 = fopen("sumIx2.txt", "w");
    FILE* fpixy = fopen("sumIxy.txt", "w");
    FILE* fpiy2 = fopen("sumIy2.txt", "w");
#endif
    for (int i = 0; i < gx.rows; i++) {
        for (int j = 0; j < gx.cols; j++) {
            float sumIx2 = 0, sumIxIy = 0, sumIy2 = 0;
            for (int ki = MAX(0, i - WINSIZE_OFLOW / 2); ki <= MIN(gx.rows - 1, i + WINSIZE_OFLOW / 2); ki++) {
                for (int kj = MAX(0, j - WINSIZE_OFLOW / 2); kj <= MIN(gx.cols - 1, j + WINSIZE_OFLOW / 2); kj++) {
                    short int tmpGx = gx.at<short int>(ki, kj);
                    short int tmpGy = gy.at<short int>(ki, kj);
                    sumIx2 += ((float)(tmpGx * tmpGx));
                    sumIy2 += ((float)(tmpGy * tmpGy));
                    sumIxIy += ((float)(tmpGx * tmpGy));
                } // end kj loop
            }     // end ki loop
            sumIx2 /= 4;
            sumIy2 /= 4;
            sumIxIy /= 4;

            int offset = 3 * (i * (gx.cols) + j);
            *(matrixG + offset) = 65.025 + sumIx2;
            *(matrixG + offset + 1) = 65.025 + sumIy2;
            *(matrixG + offset + 2) = sumIxIy;
#if DEBUG
            fprintf(fpix2, "%12.2f ", sumIx2);
            fprintf(fpixy, "%12.2f ", sumIxIy);
            fprintf(fpiy2, "%12.2f ", sumIy2);
#endif
        } // end j loop
#if DEBUG
        fprintf(fpix2, "\n");
        fprintf(fpixy, "\n");
        fprintf(fpiy2, "\n");
#endif
    } // end i loop
#if DEBUG
    fclose(fpix2);
    fclose(fpixy);
    fclose(fpiy2);
#endif
} // end constructGradientMatrix()

void findIt(cv::Mat im0, cv::Mat im1, cv::Mat nuKx, cv::Mat nuKy, cv::Mat& gt, int level) {
    for (int i = 0; i < im0.rows; i++) {
        for (int j = 0; j < im0.cols; j++) {
            float resPix;
            float indx = (float)j + nuKx.at<float>(i, j);
            float indy = (float)i + nuKy.at<float>(i, j);
            if (indx < 0 || indy < 0 || indx > im0.cols - 2 || indy > im0.rows - 2) {
                resPix = (float)im1.at<unsigned char>(i, j);
            } else {
                int indx0 = (int)indx;
                int indx1 = indx0 + 1;
                int indy0 = (int)indy;
                int indy1 = indy0 + 1;

                float ratioX = indx - (float)indx0;
                float ratioY = indy - (float)indy0;

                unsigned char i1 = im1.at<unsigned char>(indy0, indx0);
                unsigned char i2 = im1.at<unsigned char>(indy0, indx1);
                unsigned char i3 = im1.at<unsigned char>(indy1, indx0);
                unsigned char i4 = im1.at<unsigned char>(indy1, indx1);

                resPix = (1 - ratioX) * (1 - ratioY) * ((float)i1) + (ratioX) * (1 - ratioY) * ((float)i2) +
                         (1 - ratioX) * (ratioY) * ((float)i3) + (ratioX) * (ratioY) * ((float)i4);
            }

            gt.at<float>(i, j) = (float)(im0.at<unsigned char>(i, j)) - resPix;

            // gt.at<float>(i,j) = (float)(im0.at<unsigned char>(i,j) - im1.at<unsigned char>(i,j));
            int dummy = 0;
        }
    }
}

void buildMismatchVector(cv::Mat gx, cv::Mat gy, cv::Mat gt, float* vectorB, int winsize) {
    // void buildMismatchVector(cv::Mat gx, cv::Mat gy, cv::Mat im0, cv::Mat im1, cv::Mat glx, cv::Mat gly, cv::Mat
    // nuKx, cv::Mat nuKy, float *vectorB, int winsize, int level) {
    int ind = 0;
#if DEBUG
    FILE* fpixt = fopen("sumIxt.txt", "w");
    FILE* fpiyt = fopen("sumIyt.txt", "w");
#endif
    for (int i = 0; i < gx.rows; i++) {
        for (int j = 0; j < gx.cols; j++) {
            float sumIxIt = 0, sumIyIt = 0;

            for (int ki = MAX(0, i - WINSIZE_OFLOW / 2); ki <= MIN(gx.rows - 1, i + WINSIZE_OFLOW / 2); ki++) {
                for (int kj = MAX(0, j - WINSIZE_OFLOW / 2); kj <= MIN(gx.cols - 1, j + WINSIZE_OFLOW / 2); kj++) {
                    sumIxIt += ((float)gx.at<short int>(ki, kj) * gt.at<float>(ki, kj));
                    sumIyIt += ((float)gy.at<short int>(ki, kj) * gt.at<float>(ki, kj));
                }
            }

            *(vectorB + ind) = sumIxIt / 2; // 32;
            ind++;
            *(vectorB + ind) = sumIyIt / 2; // 32;
            ind++;
#if DEBUG
            fprintf(fpixt, "%12.4f ", sumIxIt);
            fprintf(fpiyt, "%12.4f ", sumIyIt);
#endif
        }
#if DEBUG
        fprintf(fpixt, "\n");
        fprintf(fpiyt, "\n");
#endif
    }
#if DEBUG
    fclose(fpixt);
    fclose(fpiyt);
#endif
}

void computeOpticalFlow(float* matrixG, float* vectorB, cv::Mat& nuKx, cv::Mat& nuKy, int rows, int cols, int winsize) {
    int ind = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int offsetG = 3 * ind;
            int offsetB = 2 * ind;
            float sumIx2 = *(matrixG + offsetG);
            float sumIy2 = *(matrixG + offsetG + 1);
            float sumIxIy = *(matrixG + offsetG + 2);
            float sumIxIt = *(vectorB + offsetB);
            float sumIyIt = *(vectorB + offsetB + 1);

            float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
            float ux, uy;
            float minEig = (sumIy2 + sumIx2 - sqrt((sumIx2 - sumIy2) * (sumIx2 - sumIy2) + 4.f * sumIxIy * sumIxIy)) /
                           (2 * WINSIZE_OFLOW * WINSIZE_OFLOW);

            if (det < 1.19209289550781250000e-7F || minEig < minEigThreshold) {
                ux = 0;
                uy = 0;
            } else {
                float divisor = 1.0 / (det);
                ux = divisor * (sumIy2 * sumIxIt - sumIxIy * sumIyIt);
                uy = divisor * (sumIx2 * sumIyIt - sumIxIy * sumIxIt);
            }
            nuKx.at<float>(i, j) += ux;
            nuKy.at<float>(i, j) += uy;
            ind++;
        }
    }
}

void updateFlow(cv::Mat& glx, cv::Mat& gly, cv::Mat nuKx, cv::Mat nuKy, int level) {
    int lvl = (level == 0) ? level : level - 1;
    int in_rows = glx.rows >> level;
    int in_cols = glx.cols >> level;
    int out_rows = glx.rows >> lvl;
    int out_cols = glx.cols >> lvl;
    float incr = (float)(in_rows - 1) / (out_rows - 1);
    for (int i = 0; i<glx.rows>> lvl; i++) {
        for (int j = 0; j<glx.cols>> lvl; j++) {
            // with interpolation
            float iflt = i * incr;
            float jflt = j * incr;
            int i0 = MAX((int)iflt, 0);
            int j0 = MAX((int)jflt, 0);
            int i1 = MIN(nuKx.rows - 1, i0 + 1);
            int j1 = MIN(nuKx.cols - 1, j0 + 1);
            float a = iflt - (float)i0;
            float b = jflt - (float)j0;
            float toAddX = (1 - a) * (1 - b) * nuKx.at<float>(i0, j0) + (1 - a) * b * nuKx.at<float>(i0, j1) +
                           a * (1 - b) * nuKx.at<float>(i1, j0) + a * b * nuKx.at<float>(i1, j1);
            float toAddY = (1 - a) * (1 - b) * nuKy.at<float>(i0, j0) + (1 - a) * b * nuKy.at<float>(i0, j1) +
                           a * (1 - b) * nuKy.at<float>(i1, j0) + a * b * nuKy.at<float>(i1, j1);
            int mul;
            if (level == lvl)
                mul = 1;
            else
                mul = 2;

            glx.at<float>(i, j) = mul * toAddX;
            gly.at<float>(i, j) = mul * toAddY;

            // without interpolation
            // glx.at<float>(i,j) += (2*glx.at<float>(i,j) + nuKx.at<float>(i>>level,j>>level));
            // gly.at<float>(i,j) += (2*gly.at<float>(i,j) + nuKy.at<float>(i>>level,j>>level));
        }
    }
}

void findIm1(cv::Mat im1Orig, cv::Mat nuKx, cv::Mat nuKy, cv::Mat& im1) {
    cv::Mat mapx(im1Orig.size(), CV_32F);
    cv::Mat mapy(im1Orig.size(), CV_32F);

    for (int i = 0; i < mapx.rows; i++) {
        for (int j = 0; j < mapx.cols; j++) {
            mapx.at<float>(i, j) = (float)j + nuKx.at<float>(i, j);
            mapy.at<float>(i, j) = (float)i + nuKy.at<float>(i, j);
        }
    }
    remap(im1Orig, im1, mapx, mapy, cv::INTER_LINEAR);
    // 	imwrite("remapped_out.png", im1);
}

void refOpticalFlow(cv::Mat imgRef, cv::Mat imgTarget, cv::Mat& glx, cv::Mat& gly) {
    /*** Create the pyramid for ref and target frames ***/
    vector<cv::Mat> imPyr0, imPyr1;

    imPyr0.push_back(imgRef);
    imPyr1.push_back(imgTarget);

    stringstream filename0;
    stringstream filename1;
    for (int i = 1; i < NUM_LEVELS; i++) {
        cv::Mat tmpIm, tmpIm0, tmpIm1;

#if DEBUG
        filename0.str("");
        filename1.str("");
        filename0 << "FileRef-" << i << ".png";
        filename1 << "FileNext-" << i << ".png";
#endif
        GaussianBlur(imPyr0[i - 1], tmpIm, cv::Size(5, 5), 1, 1, cv::BORDER_DEFAULT);
        resize(tmpIm, tmpIm0, cv::Size(0, 0), 0.5, 0.5, cv::INTER_LINEAR);
        imPyr0.push_back(tmpIm0);

        GaussianBlur(imPyr1[i - 1], tmpIm, cv::Size(5, 5), 1, 1, cv::BORDER_DEFAULT);
        resize(tmpIm, tmpIm1, cv::Size(0, 0), 0.5, 0.5, cv::INTER_LINEAR);
        imPyr1.push_back(tmpIm1);

#if DEBUG
        imwrite(filename0.str(), imPyr0[i]);
        imwrite(filename1.str(), imPyr1[i]);
#endif
    }

    /*** Find the optical flow ***/

    for (int i = NUM_LEVELS - 1; i >= 0; i--) {
        cv::Mat im0, im1;
        im0 = imPyr0[i];
        cv::Mat gx(im0.size(), CV_16S), gy(im0.size(), CV_16S);
        cout << " Level = " << i << ";\t rows = " << im0.rows << ";\t cols = " << im0.cols << endl;
        // Scharr(im0, gx, CV_16S, 1, 0);
        // Scharr(im0, gy, CV_16S, 0, 1);
        find_ix_iy(im0, gx, gy);
        // write_to_file_short(gx, "gx.txt");
        // write_to_file_short(gy, "gy.txt");
        float* matrixG = (float*)malloc(3 * (im0.rows) * (im0.cols) * sizeof(float));
        float* vectorB = (float*)malloc(2 * (im0.rows) * (im0.cols) * sizeof(float));
        constructGradientMatrix(gx, gy, WINSIZE_OFLOW, matrixG);

        cv::Mat nuKx = glx.clone(); //(im0.size(), CV_32F, Scalar::all(0));
        cv::Mat nuKy = gly.clone(); //(im0.size(), CV_32F, Scalar::all(0));

        for (int k = 0; k < NUM_ITERATIONS; k++) {
            cv::Mat gt(im0.size(), CV_32F);
            cv::Mat im1(im0.size(), CV_8U);
            // findIm1(imPyr1[i], glx, gly, nuKx, nuKy, i, im1);
            findIt(im0, imPyr1[i], nuKx, nuKy, gt, i);
            // findIt(im0, im1, glx, gly, nuKx, nuKy, gt, i);
            // write_to_file_float(gt, "gt.txt");
            buildMismatchVector(gx, gy, gt, vectorB, WINSIZE_OFLOW);
            // buildMismatchVector(gx, gy, im0, imPyr1[i], glx, gly, nuKx, nuKy, vectorB, WINSIZE_OFLOW, i);
            computeOpticalFlow(matrixG, vectorB, nuKx, nuKy, im0.rows, im0.cols, WINSIZE_OFLOW);
            // write_to_file_float(nuKx, "ux.txt");
            // write_to_file_float(nuKy, "uy.txt");
            medianBlur(nuKx, nuKx, MEDIAN_FILTER_SIZE);
            medianBlur(nuKy, nuKy, MEDIAN_FILTER_SIZE);
            gt.release();
            im1.release();
        } // end NUM_ITERATIONS loop
        updateFlow(glx, gly, nuKx, nuKy, i);
        // write_result_to_image (imPyr1[i], glx, gly, i);
        // write_to_file_float(glx, "glx.txt");
        // write_to_file_float(gly, "gly.txt");

        free(matrixG);
        free(vectorB);
        nuKx.release();
        nuKy.release();
        gx.release();
        gy.release();
    } // end NUM_LEVELS loop

    // write_result_to_image (imgTarget, glx, gly, 0);
    cv::Mat im_dummy;
    findIm1(imgTarget, glx, gly, im_dummy);
}