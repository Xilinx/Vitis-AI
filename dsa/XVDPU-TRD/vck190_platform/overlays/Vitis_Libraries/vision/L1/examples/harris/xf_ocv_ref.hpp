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

#ifndef _XF_OCV_REF_HPP_
#define _XF_OCV_REF_HPP_

#include "xf_harris_config.h"

using namespace cv;
using namespace std;

typedef float NMSTYPE;

bool OCVFindMaximum2(NMSTYPE* t1, NMSTYPE* t2, NMSTYPE* m1, NMSTYPE* b1, NMSTYPE* b2) {
    bool Max = false;
    NMSTYPE maxval = m1[2];
    if ((maxval > t1[2]) && (maxval > t2[1]) && (maxval > t2[2]) && (maxval > t2[3]) && (maxval > m1[0]) &&
        (maxval > m1[1]) && (maxval > m1[3]) && (maxval > m1[4]) && (maxval > b1[1]) && (maxval > b1[2]) &&
        (maxval > b1[3]) && (maxval > b2[2]))
        Max = true;
    return Max;
}

void OCVMaxSuppression2(cv::Mat& src, cv::Mat& dst) {
    int i, j, k;
    int ii, jj;
    NMSTYPE t1[5], t2[5];
    NMSTYPE m1[5];
    NMSTYPE b1[5], b2[5];
    bool result;

    /*			Process zeroth row			*/
    i = 0;
    for (j = 0; j < src.cols; j++) {
        if (j < 2) {
            for (k = 0; k < 5; k++) {
                t1[k] = 0;
            }
            for (k = 0; k < 5; k++) {
                t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    b2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b2[k] = 0;
            }
        } else if ((j >= 2) && (j < src.cols - 2)) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                m1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                b1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = src.at<NMSTYPE>(ii, jj);
            }
        } else if (j >= src.cols - 2) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    b2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b2[k] = 0;
            }
        }
        result = OCVFindMaximum2(t1, t2, m1, b1, b2);
        dst.at<uchar>(i, j) = result ? 255 : 0;
    }
    /*			Process second row			*/
    i = 1;
    for (j = 0; j < src.cols; j++) {
        if (j < 2) {
            for (k = 0; k < 5; k++) {
                t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    t2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    b2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b2[k] = 0;
            }
        } else if ((j >= 2) && (j < src.cols - 2)) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                t2[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                m1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                b1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = src.at<NMSTYPE>(ii, jj);
            }
        } else if (j >= src.cols - 2) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    b2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b2[k] = 0;
            }
        }
        result = OCVFindMaximum2(t1, t2, m1, b1, b2);
        dst.at<uchar>(i, j) = result ? 255 : 0;
    }

    for (i = 2; i < src.rows - 2; i++) {
        for (j = 0; j < src.cols; j++) {
            if (j < 2) {
                for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                    if (jj >= 0)
                        t1[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        t1[k] = 0;
                }
                for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                    if (jj >= 0)
                        t2[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        t2[k] = 0;
                }
                for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                    if (jj >= 0)
                        m1[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        m1[k] = 0;
                }
                for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                    if (jj >= 0)
                        b1[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        b1[k] = 0;
                }
                for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                    if (jj >= 0)
                        b2[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        b2[k] = 0;
                }
            } else if ((j >= 2) && (j < src.cols - 2)) {
                for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                    t1[k] = src.at<NMSTYPE>(ii, jj);
                }
                for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                    t2[k] = src.at<NMSTYPE>(ii, jj);
                }
                for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                }
                for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                }
                for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                    b2[k] = src.at<NMSTYPE>(ii, jj);
                }
            } else if (j >= src.cols - 2) {
                for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                    if (jj < src.cols)
                        t1[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        t1[k] = 0;
                }
                for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                    if (jj < src.cols)
                        t2[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        t2[k] = 0;
                }
                for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                    if (jj < src.cols)
                        m1[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        m1[k] = 0;
                }
                for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                    if (jj < src.cols)
                        b1[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        b1[k] = 0;
                }
                for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                    if (jj < src.cols)
                        b2[k] = src.at<NMSTYPE>(ii, jj);
                    else
                        b2[k] = 0;
                }
            }

            result = OCVFindMaximum2(t1, t2, m1, b1, b2);
            dst.at<uchar>(i, j) = result ? 255 : 0;
        }
    }
    /*			Process zeroth row			*/
    i = src.rows - 2;
    for (j = 0; j < src.cols; j++) {
        if (j < 2) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    t1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    t2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = 0;
            }
        } else if ((j >= 2) && (j < src.cols - 2)) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                t1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                t2[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                m1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                b1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = 0;
            }
        } else if (j >= src.cols - 2) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    t1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    t2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    b1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = 0;
            }
        }
        result = OCVFindMaximum2(t1, t2, m1, b1, b2);
        dst.at<uchar>(i, j) = result ? 255 : 0;
    }
    /*			Process second row			*/
    i = src.rows - 1;
    for (j = 0; j < src.cols; j++) {
        if (j < 2) {
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    t1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    t2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj >= 0)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = 0;
            }
        } else if ((j >= 2) && (j < src.cols - 2)) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                t1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                t2[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                m1[k] = src.at<NMSTYPE>(ii, jj);
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = 0;
            }
        } else if (j >= src.cols - 2) {
            for (k = 0, ii = i - 2, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    t1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t1[k] = 0;
            }
            for (k = 0, ii = i - 1, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    t2[k] = src.at<NMSTYPE>(ii, jj);
                else
                    t2[k] = 0;
            }
            for (k = 0, ii = i, jj = j - 2; k < 5; k++, jj++) {
                if (jj < src.cols)
                    m1[k] = src.at<NMSTYPE>(ii, jj);
                else
                    m1[k] = 0;
            }
            for (k = 0, ii = i + 1, jj = j - 2; k < 5; k++, jj++) {
                b1[k] = 0;
            }
            for (k = 0, ii = i + 2, jj = j - 2; k < 5; k++, jj++) {
                b2[k] = 0;
            }
        }
        result = OCVFindMaximum2(t1, t2, m1, b1, b2);
        dst.at<uchar>(i, j) = result ? 255 : 0;
    }
}

bool OCVFindMaximum1(
    NMSTYPE t0, NMSTYPE t1, NMSTYPE t2, NMSTYPE m0, NMSTYPE m1, NMSTYPE m2, NMSTYPE b0, NMSTYPE b1, NMSTYPE b2) {
    bool Max = false;
    if (m1 > t1 && m1 > m0 && m1 > m2 && m1 > b1) Max = true;
    return Max;
}

/*				Non maximum suppression				*/
void OCVMaxSuppression1(cv::Mat& src, cv::Mat& dst) {
    int i, j;
    NMSTYPE t0, t1, t2;
    NMSTYPE m0, m1, m2;
    NMSTYPE b0, b1, b2;
    bool result;

    /*			First row			*/
    i = 0;
    for (j = 0; j < src.cols; j++) {
        if (j == 0) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = 0;
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = src.at<NMSTYPE>(i + 1, j + 1);
        } else if ((j > 0) && (j < src.cols - 1)) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = src.at<NMSTYPE>(i + 1, j - 1);
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = src.at<NMSTYPE>(i + 1, j + 1);
        } else if (j == src.cols - 1) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = 0;
            b0 = src.at<NMSTYPE>(i + 1, j - 1);
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = 0;
        }
        result = OCVFindMaximum1(t0, t1, t2, m0, m1, m2, b0, b1, b2);
        dst.at<uchar>(i, j) = result ? 255 : 0;
    }
    for (i = 1; i < src.rows - 1; i++) {
        for (j = 0; j < src.cols; j++) {
            if (j == 0) {
                t0 = 0;
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = src.at<NMSTYPE>(i - 1, j + 1);
                m0 = 0;
                m1 = src.at<NMSTYPE>(i, j);
                m2 = src.at<NMSTYPE>(i, j + 1);
                b0 = 0;
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = src.at<NMSTYPE>(i + 1, j + 1);
            } else if ((j > 0) && (j < src.cols - 1)) {
                t0 = src.at<NMSTYPE>(i - 1, j - 1);
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = src.at<NMSTYPE>(i - 1, j + 1);
                m0 = src.at<NMSTYPE>(i, j - 1);
                m1 = src.at<NMSTYPE>(i, j);
                m2 = src.at<NMSTYPE>(i, j + 1);
                b0 = src.at<NMSTYPE>(i + 1, j - 1);
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = src.at<NMSTYPE>(i + 1, j + 1);
            } else if (j == src.cols - 1) {
                t0 = src.at<NMSTYPE>(i - 1, j - 1);
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = 0;
                m0 = src.at<NMSTYPE>(i, j - 1);
                m1 = src.at<NMSTYPE>(i, j);
                m2 = 0;
                b0 = src.at<NMSTYPE>(i + 1, j - 1);
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = 0;
            }
            result = OCVFindMaximum1(t0, t1, t2, m0, m1, m2, b0, b1, b2);
            dst.at<uchar>(i, j) = result ? 255 : 0;
        }
    }
    /*			Last row			*/
    i = src.rows - 1;
    for (j = 1; j < src.cols - 1; j++) {
        t0 = src.at<NMSTYPE>(i - 1, j - 1);
        t1 = src.at<NMSTYPE>(i - 1, j);
        t2 = src.at<NMSTYPE>(i - 1, j + 1);
        m0 = src.at<NMSTYPE>(i, j - 1);
        m1 = src.at<NMSTYPE>(i, j);
        m2 = src.at<NMSTYPE>(i, j + 1);
        b0 = 0;
        b1 = 0;
        b2 = 0;
        result = OCVFindMaximum1(t0, t1, t2, m0, m1, m2, b0, b1, b2);
        dst.at<uchar>(i, j) = result ? 255 : 0;
    }
}

void ocv_ref(cv::Mat img_gray, cv::Mat& ocv_out_img, float Th) {
    cv::Mat gradx, grady;
    cv::Mat gradx2, grady2, gradxy;
    cv::Mat gradx2g, grady2g, gradxyg;
    cv::Mat x2y2, xy, mtrace;
    cv::Mat dst, ocvthresh_img;

    /*****	Opencv Pipeline *****/
    // Step one: Apply gradient
    cv::Sobel(img_gray, gradx, CV_32FC1, 1, 0, FILTER_WIDTH, 1, 0, cv::BORDER_CONSTANT);
    cv::Sobel(img_gray, grady, CV_32FC1, 0, 1, FILTER_WIDTH, 1, 0, cv::BORDER_CONSTANT);
    // Step Two: Calculate gx^2, gy^2, gx*gy
    pow(gradx, 2.0, gradx2);
    pow(grady, 2.0, grady2);
    multiply(gradx, grady, gradxy);
    // Step Three: Apply boxfilter
    cv::boxFilter(gradx2, gradx2g, -1, cv::Size(BLOCK_WIDTH, BLOCK_WIDTH));
    cv::boxFilter(grady2, grady2g, -1, cv::Size(BLOCK_WIDTH, BLOCK_WIDTH));
    cv::boxFilter(gradxy, gradxyg, -1, cv::Size(BLOCK_WIDTH, BLOCK_WIDTH));

    multiply(gradx2g, grady2g, x2y2);
    multiply(gradxyg, gradxyg, xy);
    pow((gradx2g + grady2g), 2.0, mtrace);

    // Step Four: Compute score
    dst.create(img_gray.rows, img_gray.cols, CV_32FC1);
    ocvthresh_img.create(img_gray.rows, img_gray.cols, dst.depth());

    float sum = 0;
    for (int j = 0; j < img_gray.rows; j++) {
        for (int i = 0; i < img_gray.cols; i++) {
            float v1 = x2y2.at<float>(j, i);
            float v2 = xy.at<float>(j, i);
            float v3 = mtrace.at<float>(j, i);
            float temp1 = (v1 - v2) - (0.04 * v3);
            dst.at<float>(j, i) = temp1;

            if (temp1 > 0.0) sum += temp1;
        }
    }
    float meanval = sum / (img_gray.rows * img_gray.cols);

    // Step five: Apply Threshold
    for (int i = 0; i < img_gray.rows; i++) {
        for (int j = 0; j < img_gray.cols; j++) {
            float pix = dst.at<NMSTYPE>(i, j);
            if (pix > Th) {
                ocvthresh_img.at<NMSTYPE>(i, j) = pix;
            } else {
                ocvthresh_img.at<NMSTYPE>(i, j) = 0;
            }
        }
    }

    // Step six: Find Non maxima supppression
    if (NMS_RADIUS == 1) {
        OCVMaxSuppression1(ocvthresh_img, ocv_out_img);
    } else if (NMS_RADIUS == 2) {
        OCVMaxSuppression2(ocvthresh_img, ocv_out_img);
    }

    gradx.~Mat();
    grady.~Mat();
    gradx2.~Mat();
    grady2.~Mat();
    gradxy.~Mat();
    gradx2g.~Mat();
    grady2g.~Mat();
    gradxyg.~Mat();
    x2y2.~Mat();
    xy.~Mat();
    mtrace.~Mat();
    dst.~Mat();
    ocvthresh_img.~Mat();
    /**********		End of Opencv Pipeline		***********/
}

#endif
