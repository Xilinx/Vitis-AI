#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "OpenCVUtils.hpp"
#include "RectanglesSingleImage.hpp"

#ifndef __XCODE__
#include <cv.h>
#include <highgui.h>
#endif

using std::string;
using std::vector;
using std::ifstream;
using std::stringstream;

RectanglesSingleImage::RectanglesSingleImage(string fName) : RegionsSingleImage(fName)
{
}

RectanglesSingleImage::RectanglesSingleImage(IplImage *I) : RegionsSingleImage(I)
{
}

RectanglesSingleImage::~RectanglesSingleImage()
{
}

void RectanglesSingleImage::read(string rectFile)
{
    
    ifstream fin(rectFile.c_str());
    if(fin.is_open()){
	double x,y,w,h, sc;
	
	while(fin >> x >> y >> w >> h >> sc){
	    vector<double> *r = new vector<double>(5);
	    double myarray [] = {x,y,w,h,sc};
	    r->insert (r->begin(), myarray, myarray+5);
	    RectangleR *rect = new RectangleR(NULL, r);
	    delete(r);
	    list->push_back( (Region *)rect );
	}
    }else{
	std::cerr << "Could not open file " << rectFile << std::endl;
	assert(false);
    }
    fin.close();
}

void RectanglesSingleImage::read(ifstream &fin, int n)
{
	
    for(int i=0; i<n; i++){
	double x,y,w,h, sc;
	
	string line;
	getline(fin, line);
	stringstream ss(line);
	ss >> x >> y >> w >> h >> sc;

	vector<double> *r = new vector<double>(5);
	double myarray [] = {x,y,w,h, sc};
	r->insert (r->begin(), myarray, myarray+5);
	RectangleR *rect = new RectangleR(NULL, r);
	delete(r);
	list->push_back( (Region *)rect );
    }
}

void RectanglesSingleImage::show(){
    IplImage *mask = cvCreateImage(cvGetSize(im), im->depth, im->nChannels);
    cvCopy(im, mask, 0);
    for(unsigned int i=0; i<list->size(); i++){
	((RectangleR *)(list->at(i)))->display(mask, CV_RGB(255,0,0), 3, NULL);
    }

    showImage("Rectangles", mask);
    cvReleaseImage(&mask);
}
