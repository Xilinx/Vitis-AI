#ifndef __RECTANGLESSINGLEIMAGE_HPP__
#define __RECTANGLESSINGLEIMAGE_HPP__

#include <string>
#include <vector>
#include <fstream>

#ifndef __XCODE__
#include <cv.h>
#endif
#include "RegionsSingleImage.hpp"
#include "RectangleR.hpp"

/**
 * Specifies a set of rectangular regions for an image 
 * */
class RectanglesSingleImage : public RegionsSingleImage{

    public:
	/// Constructor: read an image from a file
	RectanglesSingleImage(std::string);
	/// Constructor: intialize the image for this set of ellipses as I
	RectanglesSingleImage(IplImage *);
	/// Destructor
	~RectanglesSingleImage();
	/// Read the annotaion from the file fName 
	virtual void read(std::string);
	/// Read N annotaion from the file stream fs 
	virtual void read(std::ifstream &, int);
	/// Display all ellipses 
	virtual void show();
};

#endif
