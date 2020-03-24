#ifndef __REGION_HPP__
#define __REGION_HPP__

#define REGION_MASK_VALUE 10

#include "common.hpp"

#ifdef __XCODE__
#include <OpenCV/OpenCV.h>
#else
#include <cv.h>
#endif

/**
 * Abstract class for the specification of a region
 *  */
class Region{
    private:
	/// Flag to specify if this region should be used
	bool valid;

    public:
	/// Image used for display and set operations
	IplImage *mask;
	/// Score assigned by an external detector
	double detScore;
	/// Constructor
	Region(IplImage *I);
	/// Destructor
	~Region();

	/// Returns if the region is valid for use
	bool isValid();
	/// Assigns the validity flag
	void setValid(bool);
	/// Computes the set-intersection between this->mask and r->mask 
	double setIntersect(Region *r);
	/// Computes the set-union between this->mask and r->mask 
	double setUnion(Region *r);
	/// Display this region -- Not implemented in this abstract class 
	virtual IplImage *display(IplImage *, CvScalar color, int lineWidth, const char *text) =0;
};

#endif

