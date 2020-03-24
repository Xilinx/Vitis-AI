#ifndef __OPENCVUTILS_HPP__
#define __OPENCVUTILS_HPP__

#include "OpenCVUtils.hpp"

using namespace std;

void matPrint(string s, const CvArr *M){
	assert(M != NULL);
		
	if(!s.empty())
		cerr << s;

	CvTypeInfo *info = cvTypeOf(M);
	if(!strcmp(info->type_name, CV_TYPE_NAME_IMAGE)){
		CvScalar s;
		IplImage *I = (IplImage *)M;
		for(int i=0; i< I->height; i++){
			for(int j=0; j< I->width; j++){
				s = cvGet2D(I,i,j);
				cerr << s.val[0] << " ";
			}
			cerr << endl;
		}
	}else if(!strcmp(info->type_name, CV_TYPE_NAME_MAT)){
		CvMat *M1 = (CvMat *)M;
		for(int i=0; i< M1->height; i++){
			for(int j=0; j< M1->width; j++)
				cerr << cvmGet(M1, i, j) << " ";
			cerr << endl;
		}
	}else{
		assert(false);
	}
}

void matRotate(const CvArr * src, CvArr * dst, double angle){
	float m[6];
	//double factor = (cos(angle*CV_PI/180.) + 1.1)*3;
	double factor = 1;
	CvMat M = cvMat( 2, 3, CV_32F, m );
	int w = ((CvMat *)src)->width;
	int h = ((CvMat *)src)->height;

	m[0] = (float)(factor*cos(-angle*CV_PI/180.));
	m[1] = (float)(factor*sin(-angle*CV_PI/180.));
	m[2] = (w-1)*0.5f;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = (h-1)*0.5f;

	cvGetQuadrangleSubPix( src, dst, &M);
}

void matCopyStuffed(const CvArr *src, CvArr *dst){

	// TODO: get a flag for default value
	//double tMin, tMax;
	//cvMinMaxLoc(src, &tMin, &tMax);
	cvSet(dst, cvScalar(0));
	CvMat *SMat = (CvMat *)src;
	CvMat *DMat = (CvMat *)dst;
	int sRow, dRow, sCol, dCol;

	if(SMat->rows >= DMat->rows){
		sRow = (SMat->rows - DMat->rows)/2;
		dRow = 0;
	}else{
		sRow = 0;
		dRow = (DMat->rows - SMat->rows)/2;
	}
	       	
	if(SMat->cols >= DMat->cols){
		sCol = (SMat->cols - DMat->cols)/2;
		dCol = 0;
	}else{
		sCol = 0;
		dCol = (DMat->cols - SMat->cols)/2;
	}

	//cerr << "src start " << sRow << " " << sCol << " dst "  << dRow << " " << dCol << endl; 
	       	
	/*
	for(int di =0; di < dRow; di++)
		for(int dj = 0; (dj < DMat->cols) && (dj < SMat->cols) ; dj++)
			cvmSet(DMat, di, dj, cvmGet(SMat, sRow, dj));
	
	for(int dj =0; dj < dCol; dj++)
		for(int di = 0; (di < DMat->rows) && (di < SMat->rows) ; di++)
			cvmSet(DMat, di, dj, cvmGet(SMat, di, sCol));
	*/
	
	for( int si = sRow, di = dRow ; (si<SMat->rows && di<DMat->rows); si++, di++)
			for( int sj = sCol, dj = dCol ; (sj<SMat->cols && dj<DMat->cols); sj++, dj++)
				cvmSet(DMat, di, dj, cvmGet(SMat, si, sj));
	
}

void matNormalize(const CvArr * src, CvArr *dst, double minVal, double maxVal){
	double tMin, tMax;
	cvMinMaxLoc(src, &tMin, &tMax);
	double scaleFactor = (maxVal-minVal)/(tMax-tMin);
	cvSubS(src, cvScalar(tMin), dst); 
	cvConvertScale(dst, dst, scaleFactor, minVal);
}

double matMedian(const CvArr *M){

	int starti=0, startj=0, height, width;
	CvTypeInfo *info = cvTypeOf(M);
	if(!strcmp(info->type_name, CV_TYPE_NAME_IMAGE)){
		CvRect r = cvGetImageROI((IplImage *)M);
		height = r.height;
		width = r.width;
		startj = r.x;
		starti = r.y;
	}else if(!strcmp(info->type_name, CV_TYPE_NAME_MAT)){
		height = ((CvMat *)M)->height;
		width = ((CvMat *)M)->width;
	}else{
		assert(false);
	}
	
	// push elements into a vector
	vector<double> v;
	for(int i=0; i< height; i++)
		for(int j=0; j<width; j++){
			v.push_back(cvGet2D(M,i,j).val[0]);
		}
	
	// sort the vector and return the median element
	sort(v.begin(), v.end());
	
	return *(v.begin() + v.size()/2);
}

void showImage(string title, const CvArr *M){
	const char *s = title.c_str();
	cvNamedWindow(s, 0);
	cvMoveWindow(s, 100, 400);
	cvShowImage(s, M);
	cvWaitKey(0);
	cvDestroyWindow(s);
}

// like imagesc
void showImageSc(string title, const CvArr *M, int height, int width){
	const char *s = title.c_str();
	IplImage *I1;
	
	CvTypeInfo *info = cvTypeOf(M);
	if(!strcmp(info->type_name, CV_TYPE_NAME_IMAGE)){
		I1 = (IplImage *)M;
	}else if(!strcmp(info->type_name, CV_TYPE_NAME_MAT)){
		CvMat *M2 = cvCloneMat((CvMat *)M);
		matNormalize(M, M2, 0, 255);
		double tMin, tMax;
		cvMinMaxLoc(M2, &tMin, &tMax);
		I1 = cvCreateImage(cvGetSize(M2), IPL_DEPTH_8U,1);
		cvConvertScale(M2, I1);
	}else{
		assert(false);
	}

	IplImage *I = cvCreateImage(cvSize(height, width), I1->depth,1);
	cvResize(I1, I);
	cvNamedWindow(s, 0);
	cvMoveWindow(s, 100, 400);
	cvShowImage(s, I);
	cvWaitKey(0);
	cvDestroyWindow(s);
	cvReleaseImage(&I);
	cvReleaseImage(&I1);
}

IplImage *readImage(const char *fileName, int useColorImage){

#ifdef _WIN32
	IplImage *img = cvLoadImage(fileName, useColorImage);
#else
	// check the extension for jpg files; OpenCV has issues with reading jpg files.
	int randInt = rand();
	char randIntStr[128];
	sprintf(randIntStr, "%d", randInt);

	string tmpPPMFile("cacheReadImage");
	tmpPPMFile += randIntStr;
	tmpPPMFile += ".ppm";

	string sysCommand = "convert ";
	sysCommand += fileName;
	sysCommand += " " + tmpPPMFile;
	system(sysCommand.c_str());

	IplImage *img = cvLoadImage(tmpPPMFile.c_str(), useColorImage);
	if(img == NULL)
	{
	  cerr << " Could not read image" << endl;
	}

	sysCommand = "rm -f ";
	sysCommand += tmpPPMFile;
	system(sysCommand.c_str());
#endif
	return img;
}

#endif
