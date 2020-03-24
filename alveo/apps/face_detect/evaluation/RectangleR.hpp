#ifndef __RECTANGLE_HPP__
#define __RECTANGLE_HPP__

#include "Region.hpp"
#include <vector>

#ifndef __XCODE__
#include <cv.h>
#endif

/**
 * Specification of a rectangular region
 *  */
class RectangleR : public Region{
    private:
	/// x-position of the left-top corner
	double x;
	/// y-position of the left-top corner
	double y;
	/// width 
	double w;
	/// height 
	double h;
    public:
	/// Constructor
	RectangleR(IplImage *, std::vector<double> *);
	/// Method to add this rectangle of a given color and 
	/// line width to an image. If the 
	/// last parameter is not NULL, display the text also.
	virtual IplImage *display(IplImage *, CvScalar color, int lineWidth, const char *text);
};

#endif
