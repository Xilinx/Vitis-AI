#ifndef __ELLIPSER_HPP__
#define __ELLIPSER_HPP__

#include "common.hpp"

#include <vector>
#ifndef __XCODE__
#include <cv.h>
#endif

#include "Region.hpp"
/**
 * Specification of an elliptical region
 *  */
class EllipseR : public Region{
    private:
	/// x-position of the center
	double cx;
	/// y-position of the center
	double cy;
	/// orientation of the major axis
	double angle;
	/// half-length of the major axis
	double ra;
	/// half-length of the minor axis
	double rb;
    public:
	/// Constructor
	EllipseR(IplImage *, std::vector<double> *);
	/// Method to add this ellipse of a given color and 
	/// line width to an image. If the 
	/// last parameter is not NULL, display the text also.
	virtual IplImage *display(IplImage *I, CvScalar color, int lineWidth, const char *text);
};

#endif
