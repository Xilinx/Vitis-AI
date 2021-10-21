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

#ifndef __OBJDET_REFERENCE__
#define __OBJDET_REFERENCE__

#if __SDSOC
#undef __ARM_NEON__
#undef __ARM_NEON
#include "opencv2/core/core.hpp"
#include "opencv2/core/operations.hpp"
#define __ARM_NEON__
#define __ARM_NEON
#else
#include "opencv2/core/core.hpp"
#include "opencv2/core/operations.hpp"
#endif

using namespace cv;
using namespace std;

//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////
struct AURHOGDescriptor {
   public:
    enum { L2Hys = 0 };
    enum { DEFAULT_NLEVELS = 64 };

    AURHOGDescriptor()
        : winSize(64, 128),
          blockSize(16, 16),
          blockStride(8, 8),
          cellSize(8, 8),
          nbins(9),
          derivAperture(1),
          winSigma(-1),
          histogramNormType(AURHOGDescriptor::L2Hys),
          L2HysThreshold(0.2),
          gammaCorrection(true),
          nlevels(AURHOGDescriptor::DEFAULT_NLEVELS) {}

    AURHOGDescriptor(cv::Size _winSize,
                     cv::Size _blockSize,
                     cv::Size _blockStride,
                     cv::Size _cellSize,
                     int _nbins,
                     int _derivAperture = 1,
                     double _winSigma = -1,
                     int _histogramNormType = AURHOGDescriptor::L2Hys,
                     double _L2HysThreshold = 0.2,
                     bool _gammaCorrection = false,
                     int _nlevels = AURHOGDescriptor::DEFAULT_NLEVELS)
        : winSize(_winSize),
          blockSize(_blockSize),
          blockStride(_blockStride),
          cellSize(_cellSize),
          nbins(_nbins),
          derivAperture(_derivAperture),
          winSigma(_winSigma),
          histogramNormType(_histogramNormType),
          L2HysThreshold(_L2HysThreshold),
          gammaCorrection(_gammaCorrection),
          nlevels(_nlevels) {}

    virtual ~AURHOGDescriptor() {}

    size_t AURgetDescriptorSize() const;
    double AURgetWinSigma() const;

    virtual void AURcompute(const cv::Mat& img,
                            std::vector<float>& descriptors,
                            cv::Size winStride = cv::Size(),
                            cv::Size padding = cv::Size(),
                            const std::vector<cv::Point>& locations = std::vector<cv::Point>()) const;

    virtual void AURcomputeGradient(const cv::Mat& img,
                                    cv::Mat& grad,
                                    cv::Mat& angleOfs,
                                    cv::Size paddingTL = cv::Size(),
                                    cv::Size paddingBR = cv::Size()) const;

    cv::Size winSize;
    cv::Size blockSize;
    cv::Size blockStride;
    cv::Size cellSize;
    int nbins;
    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    std::vector<float> svmDetector;
    int nlevels;
};

// column major to row major converter
//(Descriptor output from the OpenCV must be transposed from Column major to Row major format for the testing purpose)
void cmToRmConv(std::vector<float>& descriptorsValues, float* OCVdesc, int total_no_of_windows) {
    // various parameters used for testing purpose
    int novcpb = XF_BLOCK_HEIGHT / XF_CELL_HEIGHT;
    int nohcpb = XF_BLOCK_WIDTH / XF_CELL_WIDTH;
    int nob_tb = XF_NO_OF_BINS * nohcpb * novcpb;
    int novbpw_tb = ((XF_WIN_HEIGHT / XF_CELL_HEIGHT) - 1), nohbpw_tb = ((XF_WIN_WIDTH / XF_CELL_WIDTH) - 1);
    int no_of_desc_per_window = nob_tb * novbpw_tb * nohbpw_tb;

    int arr_idx = 0, row_idx = 0, col_idx = 0, win_shift_val = 0;

    for (int w = 0; w < total_no_of_windows; w++) {
        arr_idx = 0, row_idx = 0, col_idx = 0;
        for (int i = 0; i < (novbpw_tb * nohbpw_tb); i++) {
            arr_idx = (((row_idx * nohbpw_tb) * nob_tb) + (col_idx * nob_tb));

            for (int j = 0; j < nob_tb; j++) {
                int out_idx = win_shift_val + arr_idx + j;
                int in_idx = win_shift_val + (i * nob_tb) + j;
                OCVdesc[win_shift_val + arr_idx + j] = descriptorsValues[win_shift_val + (i * nob_tb) + j];
            }

            row_idx++;
            if (row_idx == novbpw_tb) {
                row_idx = 0;
                col_idx++;
            }
        }
        win_shift_val += (novbpw_tb * nohbpw_tb * nob_tb);
    }
}

/****************************************************************************************\
      The code below is implementation of HOG (Histogram-of-Oriented Gradients)
      descriptor and object detection, introduced by Navneet Dalal and Bill Triggs.

      The computed feature vectors are compatible with the
      INRIA Object Detection and Localization Toolkit
      (http://pascal.inrialpes.fr/soft/olt/)
\****************************************************************************************/

//! various border interpolation methods
enum {
    AUR_BORDER_REPLICATE,
    AUR_BORDER_CONSTANT,
    AUR_BORDER_REFLECT,
    AUR_BORDER_WRAP,
    AUR_BORDER_REFLECT_101,
    AUR_BORDER_REFLECT101 = AUR_BORDER_REFLECT_101,
    AUR_BORDER_TRANSPARENT,
    AUR_BORDER_DEFAULT = AUR_BORDER_REFLECT_101,
    AUR_BORDER_ISOLATED
};

//! 1D interpolation function: returns coordinate of the "donor" pixel for the specified location p.

int AURborderInterpolate(int p, int len, int borderType) {
    if ((unsigned)p < (unsigned)len)
        ;
    else if (borderType == AUR_BORDER_REPLICATE)
        p = p < 0 ? 0 : len - 1;
    else if (borderType == AUR_BORDER_REFLECT || borderType == AUR_BORDER_REFLECT_101) {
        int delta = borderType == AUR_BORDER_REFLECT_101;
        if (len == 1) return 0;
        do {
            if (p < 0)
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        } while ((unsigned)p >= (unsigned)len);
    } else if (borderType == AUR_BORDER_WRAP) {
        if (p < 0) p -= ((p - len + 1) / len) * len;
        if (p >= len) p %= len;
    } else if (borderType == AUR_BORDER_CONSTANT)
        p = -1;

    return p;
}

size_t AURHOGDescriptor::AURgetDescriptorSize() const {
    return (size_t)nbins * (blockSize.width / cellSize.width) * (blockSize.height / cellSize.height) *
           ((winSize.width - blockSize.width) / blockStride.width + 1) *
           ((winSize.height - blockSize.height) / blockStride.height + 1);
}

double AURHOGDescriptor::AURgetWinSigma() const {
    return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height) / 8.;
}

void AURHOGDescriptor::AURcomputeGradient(
    const cv::Mat& img, cv::Mat& grad, cv::Mat& qangle, cv::Size paddingTL, cv::Size paddingBR) const {
    cv::Size gradsize(img.cols + paddingTL.width + paddingBR.width, img.rows + paddingTL.height + paddingBR.height);
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation
    cv::Size wholeSize;
    cv::Point roiofs;
    img.locateROI(wholeSize, roiofs);

    int i, x, y;
    int cn = img.channels();

    Mat_<float> _lut(1, 256);
    const float* lut = &_lut(0, 0);

    if (gammaCorrection)
        for (i = 0; i < 256; i++) _lut(0, i) = std::sqrt((float)i);
    else
        for (i = 0; i < 256; i++) _lut(0, i) = (float)i;

    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2;

    const int borderType = (int)AUR_BORDER_CONSTANT;

    for (x = -1; x < gradsize.width + 1; x++)
        xmap[x] = AURborderInterpolate(x - paddingTL.width + roiofs.x, wholeSize.width, borderType) - roiofs.x;
    for (y = -1; y < gradsize.height + 1; y++)
        ymap[y] = AURborderInterpolate(y - paddingTL.height + roiofs.y, wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width * 4);
    float* dbuf = _dbuf;
    cv::Mat Dx(1, width, CV_32F, dbuf);
    cv::Mat Dy(1, width, CV_32F, dbuf + width);
    cv::Mat Mag(1, width, CV_32F, dbuf + width * 2);
    cv::Mat Angle(1, width, CV_32F, dbuf + width * 3);

    int _nbins = nbins;
    float angleScale = (float)(_nbins / CV_PI);

    for (y = 0; y < gradsize.height; y++) {
        const uchar* imgPtr = img.data + img.step * ymap[y];
        const uchar* prevPtr = img.data + img.step * ymap[y - 1];
        const uchar* nextPtr = img.data + img.step * ymap[y + 1];

        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);

        if (cn == 1) {
            for (x = 0; x < width; x++) {
                int x1 = xmap[x];

                if (x == 0)
                    dbuf[x] = (float)(lut[imgPtr[xmap[x + 1]]] - 0);
                else if (x == (width - 1))
                    dbuf[x] = (float)(0 - lut[imgPtr[xmap[x - 1]]]);
                else
                    dbuf[x] = (float)(lut[imgPtr[xmap[x + 1]]] - lut[imgPtr[xmap[x - 1]]]);

                if (y == 0)
                    dbuf[width + x] = (float)(lut[nextPtr[x1]] - 0);
                else if (y == (gradsize.height - 1))
                    dbuf[width + x] = (float)(0 - lut[prevPtr[x1]]);
                else
                    dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        } else {
            for (x = 0; x < width; x++) {
                int x1 = xmap[x] * 3;
                float dx0, dy0, dx, dy, mag0, mag;

                const uchar* p2 = imgPtr + xmap[x + 1] * 3;
                const uchar* p0 = imgPtr + xmap[x - 1] * 3;

                // plane 1 - x-gradient
                if (x == 0)
                    dx0 = lut[p2[2]] - 0;
                else if (x == (width - 1))
                    dx0 = 0 - lut[p0[2]];
                else
                    dx0 = lut[p2[2]] - lut[p0[2]];

                // plane 1 - y-gradient
                if (y == 0)
                    dy0 = lut[nextPtr[x1 + 2]] - 0;
                else if (y == (gradsize.height - 1))
                    dy0 = 0 - lut[prevPtr[x1 + 2]];
                else
                    dy0 = lut[nextPtr[x1 + 2]] - lut[prevPtr[x1 + 2]];

                // plane 1 - magnitude
                mag0 = dx0 * dx0 + dy0 * dy0;

                // plane 2 - x-gradient
                if (x == 0)
                    dx = lut[p2[1]] - 0;
                else if (x == (width - 1))
                    dx = 0 - lut[p0[1]];
                else
                    dx = lut[p2[1]] - lut[p0[1]];

                // plane 2 - y-gradient
                if (y == 0)
                    dy = lut[nextPtr[x1 + 1]] - 0;
                else if (y == (gradsize.height - 1))
                    dy = 0 - lut[prevPtr[x1 + 1]];
                else
                    dy = lut[nextPtr[x1 + 1]] - lut[prevPtr[x1 + 1]];

                // plane 2 - magnitude
                mag = dx * dx + dy * dy;

                // plane 1 mag vs plane 2 mag
                if (mag0 < mag) {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                // plane 3 - x-gradient
                if (x == 0)
                    dx = lut[p2[0]] - 0;
                else if (x == (width - 1))
                    dx = 0 - lut[p0[0]];
                else
                    dx = lut[p2[0]] - lut[p0[0]];

                // plane 3 - y-gradient
                if (y == 0)
                    dy = lut[nextPtr[x1]] - 0;
                else if (y == (gradsize.height - 1))
                    dy = 0 - lut[prevPtr[x1]];
                else
                    dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];

                // plane 3 - magnitude
                mag = dx * dx + dy * dy;

                if (mag0 < mag) {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dbuf[x] = dx0;
                dbuf[x + width] = dy0;
            }
        }

        cartToPolar(Dx, Dy, Mag, Angle, false);

        for (x = 0; x < width; x++) {
            float mag = dbuf[x + width * 2], angle = dbuf[x + width * 3] * angleScale - 0.5f;
            int hidx = cvFloor(angle);
            angle -= hidx;
            gradPtr[x * 2] = mag * (1.f - angle);
            gradPtr[x * 2 + 1] = mag * angle;

            if (hidx < 0)
                hidx += _nbins;
            else if (hidx >= _nbins)
                hidx -= _nbins;
            assert((unsigned)hidx < (unsigned)_nbins);

            qanglePtr[x * 2] = (uchar)hidx;
            hidx++;
            hidx &= hidx < _nbins ? -1 : 0;
            qanglePtr[x * 2 + 1] = (uchar)hidx;
        }
    }
}

struct AURHOGCache {
    struct BlockData {
        BlockData() : histOfs(0), imgOffset() {}
        int histOfs;
        cv::Point imgOffset;
    };

    struct PixData {
        size_t gradOfs, qangleOfs;
        int histOfs[4];
        float histWeights[4];
        float gradWeight;
    };

    AURHOGCache();
    AURHOGCache(const AURHOGDescriptor* descriptor,
                const cv::Mat& img,
                cv::Size paddingTL,
                cv::Size paddingBR,
                bool useCache,
                cv::Size cacheStride);
    virtual ~AURHOGCache(){};
    virtual void init(const AURHOGDescriptor* descriptor,
                      const cv::Mat& img,
                      cv::Size paddingTL,
                      cv::Size paddingBR,
                      bool useCache,
                      cv::Size cacheStride);

    cv::Size windowsInImage(cv::Size imageSize, cv::Size winStride) const;
    cv::Rect getWindow(cv::Size imageSize, cv::Size winStride, int idx) const;

    const float* getBlock(cv::Point pt, float* buf);
    virtual void normalizeBlockHistogram(float* histogram) const;

    vector<PixData> pixData;
    vector<BlockData> blockData;

    bool useCache;
    vector<int> ymaxCached;
    cv::Size winSize, cacheStride;
    cv::Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    cv::Point imgoffset;
    Mat_<float> blockCache;
    Mat_<uchar> blockCacheFlags;

    cv::Mat grad, qangle;
    const AURHOGDescriptor* descriptor;
};

AURHOGCache::AURHOGCache() {
    useCache = false;
    blockHistogramSize = count1 = count2 = count4 = 0;
    descriptor = 0;
}

AURHOGCache::AURHOGCache(const AURHOGDescriptor* _descriptor,
                         const cv::Mat& _img,
                         cv::Size _paddingTL,
                         cv::Size _paddingBR,
                         bool _useCache,
                         cv::Size _cacheStride) {
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

void AURHOGCache::init(const AURHOGDescriptor* _descriptor,
                       const cv::Mat& _img,
                       cv::Size _paddingTL,
                       cv::Size _paddingBR,
                       bool _useCache,
                       cv::Size _cacheStride) {
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;

    descriptor->AURcomputeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
    imgoffset = _paddingTL;

    winSize = descriptor->winSize;
    cv::Size blockSize = descriptor->blockSize;
    cv::Size blockStride = descriptor->blockStride;
    cv::Size cellSize = descriptor->cellSize;
    int i, j, nbins = descriptor->nbins;
    int rawBlockSize = blockSize.width * blockSize.height;

    nblocks = cv::Size((winSize.width - blockSize.width) / blockStride.width + 1,
                       (winSize.height - blockSize.height) / blockStride.height + 1);
    ncells = cv::Size(blockSize.width / cellSize.width, blockSize.height / cellSize.height);
    blockHistogramSize = ncells.width * ncells.height * nbins;

    if (useCache) {
        cv::Size cacheSize((grad.cols - blockSize.width) / cacheStride.width + 1,
                           (winSize.height / cacheStride.height) + 1);
        blockCache.create(cacheSize.height, cacheSize.width * blockHistogramSize);
        blockCacheFlags.create(cacheSize);
        size_t cacheRows = blockCache.rows;
        ymaxCached.resize(cacheRows);
        for (size_t ii = 0; ii < cacheRows; ii++) ymaxCached[ii] = -1;
    }

    Mat_<float> weights(blockSize);
    float sigma = (float)descriptor->AURgetWinSigma();
    float scale = 1.f / (sigma * sigma * 2);

    for (i = 0; i < blockSize.height; i++)
        for (j = 0; j < blockSize.width; j++) {
            float di = i - blockSize.height * 0.5f;
            float dj = j - blockSize.width * 0.5f;
            float tmp_weight = std::exp(-(di * di + dj * dj) * scale);
            weights(i, j) = tmp_weight;
        }

    blockData.resize(nblocks.width * nblocks.height);
    pixData.resize(rawBlockSize * 3);

    // Initialize 2 lookup tables, pixData & blockData.
    // Here is why:
    //
    // The detection algorithm runs in 4 nested loops (at each pyramid layer):
    //  loop over the windows within the input image
    //    loop over the blocks within each window
    //      loop over the cells within each block
    //        loop over the pixels in each cell
    //
    // As each of the loops runs over a 2-dimensional array,
    // we could get 8(!) nested loops in total, which is very-very slow.
    //
    // To speed the things up, we do the following:
    //   1. loop over windows is unrolled in the HOGDescriptor::{compute|detect} methods;
    //         inside we compute the current search window using getWindow() method.
    //         Yes, it involves some overhead (function call + couple of divisions),
    //         but it's tiny in fact.
    //   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
    //         to set up gradient and histogram pointers.
    //   3. loops over cells and pixels in each cell are merged
    //       (since there is no overlap between cells, each pixel in the block is processed once)
    //      and also unrolled. Inside we use PixData[k] to access the gradient values and
    //      update the histogram
    //
    count1 = count2 = count4 = 0;
    for (j = 0; j < blockSize.width; j++)
        for (i = 0; i < blockSize.height; i++) {
            /////// debugg from here  ///////
            PixData* data = 0;
            float cellX = (j + 0.5f) / cellSize.width - 0.5f;
            float cellY = (i + 0.5f) / cellSize.height - 0.5f;
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            cellX -= icellX0;
            cellY -= icellY0;

            if ((unsigned)icellX0 < (unsigned)ncells.width && (unsigned)icellX1 < (unsigned)ncells.width) {
                if ((unsigned)icellY0 < (unsigned)ncells.height && (unsigned)icellY1 < (unsigned)ncells.height) {
                    data = &pixData[rawBlockSize * 2 + (count4++)];
                    data->histOfs[0] = (icellX0 * ncells.height + icellY0) * nbins;
                    data->histWeights[0] = (1.f - cellX) * (1.f - cellY);
                    data->histOfs[1] = (icellX1 * ncells.height + icellY0) * nbins;
                    data->histWeights[1] = cellX * (1.f - cellY);
                    data->histOfs[2] = (icellX0 * ncells.height + icellY1) * nbins;
                    data->histWeights[2] = (1.f - cellX) * cellY;
                    data->histOfs[3] = (icellX1 * ncells.height + icellY1) * nbins;
                    data->histWeights[3] = cellX * cellY;
                } else {
                    data = &pixData[rawBlockSize + (count2++)];
                    if ((unsigned)icellY0 < (unsigned)ncells.height) {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX0 * ncells.height + icellY1) * nbins;
                    data->histWeights[0] = (1.f - cellX) * cellY;
                    data->histOfs[1] = (icellX1 * ncells.height + icellY1) * nbins;
                    data->histWeights[1] = cellX * cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            } else {
                if ((unsigned)icellX0 < (unsigned)ncells.width) {
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }

                if ((unsigned)icellY0 < (unsigned)ncells.height && (unsigned)icellY1 < (unsigned)ncells.height) {
                    data = &pixData[rawBlockSize + (count2++)];
                    data->histOfs[0] = (icellX1 * ncells.height + icellY0) * nbins;
                    data->histWeights[0] = cellX * (1.f - cellY);
                    data->histOfs[1] = (icellX1 * ncells.height + icellY1) * nbins;
                    data->histWeights[1] = cellX * cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                } else {
                    data = &pixData[count1++];
                    if ((unsigned)icellY0 < (unsigned)ncells.height) {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX1 * ncells.height + icellY1) * nbins;
                    data->histWeights[0] = cellX * cellY;
                    data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            data->gradOfs = (grad.cols * i + j) * 2;
            data->qangleOfs = (qangle.cols * i + j) * 2;
            data->gradWeight = weights(i, j);
        }

    assert(count1 + count2 + count4 == rawBlockSize);
    // defragment pixData
    for (j = 0; j < count2; j++) pixData[j + count1] = pixData[j + rawBlockSize];
    for (j = 0; j < count4; j++) pixData[j + count1 + count2] = pixData[j + rawBlockSize * 2];
    count2 += count1;
    count4 += count2;

    // initialize blockData
    for (j = 0; j < nblocks.width; j++)
        for (i = 0; i < nblocks.height; i++) {
            BlockData& data = blockData[j * nblocks.height + i];
            data.histOfs = (j * nblocks.height + i) * blockHistogramSize;
            data.imgOffset = cv::Point(j * blockStride.width, i * blockStride.height);
        }
}

const float* AURHOGCache::getBlock(cv::Point pt, float* buf) {
    float* blockHist = buf;
    assert(descriptor != 0);

    cv::Size blockSize = descriptor->blockSize;
    pt += imgoffset;

    if (useCache) {
        cv::Point cacheIdx(pt.x / cacheStride.width, (pt.y / cacheStride.height) % blockCache.rows);
        if (pt.y != ymaxCached[cacheIdx.y]) {
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        blockHist = &blockCache[cacheIdx.y][cacheIdx.x * blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if (computedFlag != 0) return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;
    const float* gradPtr = (const float*)(grad.data + grad.step * pt.y) + pt.x * 2;
    const uchar* qanglePtr = qangle.data + qangle.step * pt.y + pt.x * 2;

    for (k = 0; k < blockHistogramSize; k++) blockHist[k] = 0.f;

    const PixData* _pixData = &pixData[0];

    for (k = 0; k < C1; k++) {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w = pk.gradWeight * pk.histWeights[0];
        int t_a = a[0] + a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];
        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + a[0];
        float t1 = hist[h1] + a[1];
        hist[h0] = t0;
        hist[h1] = t1;
    }

    int total = (blockSize.height >> 2) * (blockSize.width >> 2) * 8;
    int frac_1[total]; // =
                       // {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0};
    int frac_2[total]; // =
                       // {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};

    bool flag_frac = false;
    for (int fl = 0; fl < total; fl++) {
        if (fl < (total >> 2)) {
            if ((fl % (blockSize.width >> 2)) == 0) flag_frac = !flag_frac;
            frac_1[fl] = flag_frac;
        } else if (fl < ((total >> 2) * 2)) {
            frac_1[fl] = 1;
        } else if (fl < ((total >> 2) * 3)) {
            frac_1[fl] = 0;
        } else {
            if ((fl % (blockSize.width >> 2)) == 0) flag_frac = !flag_frac;
            frac_1[fl] = flag_frac;
        }
        frac_2[fl] = !frac_1[fl];
    }
    int idx = 0;
    int total_idx = 0;

    for (; k < C2; k++) {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        int t_a = a[0] + a[1];

        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight * pk.histWeights[0];
        t0 = hist[h0] + a0 * frac_1[total_idx];
        t1 = hist[h1] + a1 * frac_1[total_idx];
        hist[h0] = t0;
        hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight * pk.histWeights[1];
        t0 = hist[h0] + a0 * frac_2[total_idx];
        t1 = hist[h1] + a1 * frac_2[total_idx];
        hist[h0] = t0;
        hist[h1] = t1;

        idx++;
        total_idx++;
    }

    int fac_1 = 1;
    int fac_2 = 0;
    int fac_3 = 1;
    int fac_4 = 0;

    idx = 0;
    total_idx = 0;

    total = (blockSize.height >> 2) * (blockSize.width >> 2) * 4;
    int f1[total]; // =
                   // {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int f2[total]; // =
                   // {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0};
    int f3[total]; // =
                   // {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int f4[total]; // =
                   // {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};

    flag_frac = false;
    for (int fl = 0; fl < total; fl++) {
        if (fl < (total >> 1)) {
            if ((fl % (blockSize.width >> 2)) == 0) flag_frac = !flag_frac;
            f1[fl] = flag_frac;
            f3[fl] = !flag_frac;
            f2[fl] = 0;
            f4[fl] = 0;
        } else {
            if ((fl % (blockSize.width >> 2)) == 0) flag_frac = !flag_frac;
            f2[fl] = flag_frac;
            f4[fl] = !flag_frac;
            f1[fl] = 0;
            f3[fl] = 0;
        }
    }

    for (; k < C4; k++) {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];
        int t_a = a[0] + a[1];

        float* hist;

        hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight * pk.histWeights[0];
        t0 = hist[h0] + a0 * f1[total_idx];
        t1 = hist[h1] + a1 * f1[total_idx];
        hist[h0] = t0;
        hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight * pk.histWeights[1];
        t0 = hist[h0] + a0 * f2[total_idx];
        t1 = hist[h1] + a1 * f2[total_idx];
        hist[h0] = t0;
        hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        w = pk.gradWeight * pk.histWeights[2];
        t0 = hist[h0] + a0 * f3[total_idx];
        t1 = hist[h1] + a1 * f3[total_idx];
        hist[h0] = t0;
        hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        w = pk.gradWeight * pk.histWeights[3];
        t0 = hist[h0] + a0 * f4[total_idx];
        t1 = hist[h1] + a1 * f4[total_idx];
        hist[h0] = t0;
        hist[h1] = t1;
        total_idx++;
    }

    normalizeBlockHistogram(blockHist);

    return blockHist;
}

void AURHOGCache::normalizeBlockHistogram(float* _hist) const {
    float* hist = &_hist[0];

    size_t i, sz = blockHistogramSize;

    float sum = 0;

    for (i = 0; i < sz; i++) sum += hist[i] * hist[i];

    float tmp_scale = (std::sqrt(sum) + sz * 0.1f);
    float scale = 1.f / tmp_scale, thresh = (float)descriptor->L2HysThreshold;

    for (i = 0, sum = 0; i < sz; i++) {
        hist[i] = std::min(hist[i] * scale, thresh);
        sum += hist[i] * hist[i];
    }

    scale = 1.f / (std::sqrt(sum) + 1e-3f);

    for (i = 0; i < sz; i++) hist[i] *= scale;
}

cv::Size AURHOGCache::windowsInImage(cv::Size imageSize, cv::Size winStride) const {
    return cv::Size((imageSize.width - winSize.width) / winStride.width + 1,
                    (imageSize.height - winSize.height) / winStride.height + 1);
}

cv::Rect AURHOGCache::getWindow(cv::Size imageSize, cv::Size winStride, int idx) const {
    int nwindowsX = (imageSize.width - winSize.width) / winStride.width + 1;
    int y = idx / nwindowsX;
    int x = idx - nwindowsX * y;
    return cv::Rect(x * winStride.width, y * winStride.height, winSize.width, winSize.height);
}
template <typename Size_t>
static inline Size_t gcd1(Size_t a, Size_t b) {
    if (a < b) std::swap(a, b);
    while (b > 0) {
        Size_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

void AURHOGDescriptor::AURcompute(const cv::Mat& img,
                                  vector<float>& descriptors,
                                  cv::Size winStride,
                                  cv::Size padding,
                                  const vector<cv::Point>& locations) const {
    if (winStride == cv::Size()) winStride = cellSize;
    cv::Size cacheStride(gcd1(winStride.width, blockStride.width), gcd1(winStride.height, blockStride.height));
    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    cv::Size paddedImgSize(img.cols + padding.width * 2, img.rows + padding.height * 2);

    AURHOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if (!nwindows) nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const AURHOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = AURgetDescriptorSize();
    descriptors.resize(dsize * nwindows);

    for (size_t i = 0; i < nwindows; i++) {
        float* descriptor = &descriptors[i * dsize];

        cv::Point pt0;
        if (!locations.empty()) {
            pt0 = locations[i];
            if (pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width || pt0.y < -padding.height ||
                pt0.y > img.rows + padding.height - winSize.height)
                continue;
        } else {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - cv::Point(padding);
        }

        for (int j = 0; j < nblocks; j++) {
            const AURHOGCache::BlockData& bj = blockData[j];
            cv::Point pt = pt0 + bj.imgOffset;

            float* dst = descriptor + bj.histOfs;
            const float* src = cache.getBlock(pt, dst);
            if (src != dst)

                for (int k = 0; k < blockHistogramSize; k++) dst[k] = src[k];
        }
    }
}

#endif
