#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "EllipsesSingleImage.hpp"

#ifndef __XCODE__
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc.hpp>
#include "OpenCVUtils.hpp"
#else
#include <OpenCV/OpenCV.h>
#endif

#ifndef M_PI
#define M_PI 3.14156
#endif

using std::string;
using std::vector;
using std::ifstream;
using std::stringstream;

EllipsesSingleImage::EllipsesSingleImage(string fName) : RegionsSingleImage(fName){
}

EllipsesSingleImage::EllipsesSingleImage(IplImage *I) : RegionsSingleImage(I){
}

EllipsesSingleImage::~EllipsesSingleImage(){
}

void EllipsesSingleImage::read(string rectFile)
{
    ifstream fin(rectFile.c_str());
    if(fin.is_open()){
	double x,y,t,w,h;
	
	while(fin >> w >> h >> t >> x >> y){
	    t = (M_PI-t) *180/M_PI;
	    vector<double> *r = new vector<double>(5);
	    double myarray [] = {x,y,t,w,h};
	    r->insert (r->begin(), myarray, myarray+5);
	    EllipseR *ell = new EllipseR(NULL, r);
	    list->push_back((Region *)ell);
	    delete(r);
	}
    }
    fin.close();
}

void EllipsesSingleImage::read(ifstream &fin, int n)
{
    for(int i=0; i< n; i++){
	double x,y,t,w,h, sc;

	string line;
	getline(fin, line);
	stringstream ss(line);
	ss >> w >> h >> t >> x >> y >> sc;

	t = (M_PI-t) *180/M_PI;
	vector<double> *r = new vector<double>(6);
	double myarray [] = {x,y,t,w,h,sc};
	r->insert (r->begin(), myarray, myarray+6);
	EllipseR *ell = new EllipseR(NULL, r);
	list->push_back((Region *)ell);
	delete(r);
    }
}

void EllipsesSingleImage::show()
{
    IplImage *mask = cvCreateImage(cvGetSize(im), im->depth, im->nChannels);
    cvCopy(im, mask, 0);
    for(unsigned int i=0; i<list->size(); i++)
	mask = ((EllipseR *)(list->at(i)))->display(mask, CV_RGB(255,0,0), 3, NULL);

    showImage("Ellipses", mask);
    cvReleaseImage(&mask);
}
