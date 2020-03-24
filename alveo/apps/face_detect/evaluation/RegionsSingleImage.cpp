#include "RegionsSingleImage.hpp"

#include <iostream>
#include <vector>

using std::vector;
using std::cerr;
using std::cout;
using std::endl;

RegionsSingleImage::RegionsSingleImage(std::string fName){
#ifdef __CVLOADIMAGE_WORKING__
  im = cvLoadImage(fName.c_str(), CV_LOAD_IMAGE_COLOR);
#else
  im = readImage(fName.c_str(), CV_LOAD_IMAGE_COLOR);
#endif
  if(im == NULL)
  {
    cerr << "Could not read image from " << fName << endl;
    assert(false);
  }
  list = new std::vector<Region *>;
}

RegionsSingleImage::RegionsSingleImage(IplImage *I)
{
  assert(I != NULL);
  im = cvCreateImage(cvGetSize(I), I->depth, I->nChannels);
  cvCopy(I, im, 0);
  list = new std::vector<Region *>;
}

RegionsSingleImage::~RegionsSingleImage()
{
  if(list)
    for(unsigned int i=0; i< list->size(); i++)
      if(list->at(i))
	delete(list->at(i)); 
  delete(list);
  cvReleaseImage(&im);
}

unsigned int RegionsSingleImage::length()
{ 
  return (unsigned int)(list->size()); 
}

Region * RegionsSingleImage::get(int i)
{ 
  return list->at(i); 
}

void RegionsSingleImage::set(int i, Region *r)
{ 
  list->at(i) = r; 
}

const IplImage *RegionsSingleImage::getImage(){
  return (const IplImage *)im; 
}

std::vector<double> * RegionsSingleImage::getUniqueScores(){
  vector<double> *v = new vector<double>;
  v->reserve(list->size());
  for(unsigned int i=0; i<list->size(); i++)
    v->push_back(list->at(i)->detScore);

  sort(v->begin(), v->end());
  vector<double>::iterator uniElem = unique(v->begin(), v->end());
  v->erase(uniElem, v->end());
  return v;
}
