#ifndef __ELLIPSESSINGLEIMAGE_HPP__
#define __ELLIPSESSINGLEIMAGE_HPP__

#include <string>
#include <vector>
#include <fstream>
#include "RegionsSingleImage.hpp"
#include "EllipseR.hpp"

#ifndef __XCODE__
#include <cv.h>
#else
#include  <OpenCV/OpenCV.h>
#endif

/**
 * Specifies a set of elliptical regions for an image 
 * */
class EllipsesSingleImage : public RegionsSingleImage{

    public:
	/// Constructor: read an image from a file
	EllipsesSingleImage(std::string);
	/// Constructor: intialize the image for this set of ellipses as I
	EllipsesSingleImage(IplImage *I);
	/// Destructor
	~EllipsesSingleImage();
	/// Read the annotaion from the file fName 
	virtual void read(std::string fName);
	/// Read N annotaion from the file stream fs 
	virtual void read(std::ifstream &fs, int N);
	/// Display all ellipses 
	virtual void show();
};

#endif
