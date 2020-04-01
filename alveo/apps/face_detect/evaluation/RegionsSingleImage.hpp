#ifndef __REGIONSSINGLEIMAGE_HPP__
#define __REGIONSSINGLEIMAGE_HPP__

#include <string>
#include <vector>

#include "common.hpp"

#ifndef __XCODE__
#include <highgui.h>
#endif

#include <iostream>
#include <fstream>
#include "Region.hpp"
#include "OpenCVUtils.hpp"

using std::cout; 
using std::cerr;
using std::endl;
using std::vector;

/**
 * Abstract class that specifies a set of regions for an image 
 * */
class RegionsSingleImage{
    protected:
	/// Image associated with this set of regions
	IplImage *im;
	/// Vector to hold the list of regions
	std::vector<Region *> *list;
    public:
	/// Constructor: read an image from a file
	RegionsSingleImage(std::string fName);
	/// Constructor: intialize the image for this set of ellipses as I
	RegionsSingleImage(IplImage *I);
	/// Destructor
	~RegionsSingleImage();

	/// Read the annotaion from the file fName -- Pure virtual 
	virtual void read(std::string)=0;
	/// Read N annotaion from the file stream fs -- Pure virtual 
	virtual void read(std::ifstream &, int)=0;
	/// Display all ellipses -- Pure virtual 
	virtual void show() =0;

	/// Returns the number of regions
	unsigned int length();
	/// Returns the pointer to the i-th region
	Region *get(int i);
	/// Adds the region r at the i-th position in the list
	void set(int i, Region *r);
	/// Returns a const pointer to the image associated with this set of regions
	const IplImage *getImage();
	/// Returns the set of unique detection scores for the regions in this set
	std::vector<double> *getUniqueScores();
};

#endif
