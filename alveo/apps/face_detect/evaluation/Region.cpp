#include "Region.hpp"

//#define REGION_DEBUG

#include "OpenCVUtils.hpp"

#ifdef REGION_DEBUG
#include <iostream>
using std::cout; 
using std::cerr;
using std::endl;
#endif

Region::Region(IplImage *I){
    // TODO: make sure this image is a mask images
    mask = I;
    valid = true;
}

Region::~Region(){
    if(mask != NULL)
	cvReleaseImage(&mask);
}

void Region::setValid(bool v){
    valid = v;
}

bool Region::isValid(){
    return valid;
}

double Region::setIntersect(Region *r){
    //TODO: check if mask and r->mask are compatible
    IplImage *temp = cvCreateImage(cvGetSize(mask), mask->depth, mask->nChannels);
    cvSetZero(temp);
    cvAnd(mask, r->mask, temp);
    int nNZ = cvCountNonZero(temp);
    double areaDbl = (double) nNZ; 

#ifdef REGION_DEBUG
    showImage("Mask Intersect", temp);
    cout << "Intersect " << areaDbl << endl;
#endif
    cvReleaseImage(&temp);

    return areaDbl;
}

double Region::setUnion(Region *r){
    //TODO:i check if mask and r->mask are compatible
    IplImage *temp = cvCreateImage(cvGetSize(mask), mask->depth, mask->nChannels);
    cvSetZero(temp);
    cvOr(mask, r->mask, temp);
    int nNZ = cvCountNonZero(temp);
    double areaDbl = (double) nNZ; 

#ifdef REGION_DEBUG
    cvConvertScale(temp, temp, 255/REGION_MASK_VALUE, 0); 
    showImage("Mask Union", temp);
    cout << "Union " << areaDbl << " " << nNZ << endl;
#endif
    cvReleaseImage(&temp);

    return areaDbl;
}
