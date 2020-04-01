#ifndef __OPENCVUTILS_H_
#define __OPENCVUTILS_H_
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

#include "common.hpp"
#ifndef __XCODE__
#include "cv.h"
#include "highgui.h"
#include <opencv2/imgproc.hpp>
#else
#include <OpenCV/OpenCV.h>
#endif

// matPrint(prefixString, M)
void matPrint(std::string, const CvArr *);

double matMedian(const CvArr *M);

void matRotate(const CvArr *src, CvArr *dst, double);

void matCopyStuffed(const CvArr *src, CvArr *dst);

void matNormalize(CvArr *, CvArr *, double, double);

// showImage(titleString, M)
void showImage(std::string, const CvArr *);

// showImageSc(titleString, M), like imagesc in MATLAB
void showImageSc(std::string, const CvArr *, int width, int height);

IplImage *readImage(const char *fileName, int useColorImage);

#endif
