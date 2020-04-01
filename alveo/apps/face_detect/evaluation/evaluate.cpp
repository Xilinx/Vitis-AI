#ifndef _WIN32
#include <unistd.h>
#endif
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>
#include <limits>

#include "common.hpp"

#ifdef __XCODE__
#include <OpenCV/OpenCV.h>
#endif

#include "RegionsSingleImage.hpp"
#include "EllipsesSingleImage.hpp"
#include "RectanglesSingleImage.hpp"
#include "Matching.hpp"
#include "Results.hpp"

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;
using std::stringstream;
using std::ifstream;
using std::ofstream;

enum detFormatTypes {DET_RECTANGLE, DET_ELLIPSE, DET_PIXELS};

vector<string> *getImageList(string inFile){
  vector<string> * imNames = new vector<string>;
  std::ifstream fin(inFile.c_str());
  if(fin.is_open()){
    string s;
    while(fin >> s){
      imNames->push_back(s);
    } 
    fin.close();
    return imNames;
  }

  cerr << "Invalid list file " << inFile << endl;
  return NULL;
}

void printUsage(){
  cout << "./evaluate [OPTIONS]" << endl;
  cout << "   -h              : print usage" << endl;
  cout << "   -a fileName     : file with face annotations (default: ellipseList.txt)" << endl;
  cout << "   -d fileName     : file with detections (default: faceList.txt)" << endl;
  cout << "   -f format       : representation of faces in the detection file (default: 0)" << endl;
  cout << "                   : [ 0 (rectangle), 1 (ellipse) or  2 (pixels) ]" << endl;
  cout << "   -i dirName      : directory where the original images are stored (default: ~/scratch/Data/facesInTheWild/)" << endl;
  cout << "   -l fileName     : file with list of images to be evaluated (default: temp.txt)" << endl;
  cout << "   -r fileName     : prefix for files to store the ROC curves (default: temp)" << endl;
}


int main(int argc, char *argv[]){

#ifndef _WIN32
  int c;
  opterr = 0;
#endif

  // default values for different command-line arguments
#ifdef _WIN32
  string baseDir = "F:/scratch/Data/facesInTheWild/";
  string listFile = "F:/scratch/Data/detectionResults/FDDB/imList.txt";
  string detFile = "F:/scratch/Data/detectionResults/FDDB/MikolajczykDets.txt";
  string annotFile = "F:/scratch/Data/detectionResults/FDDB/ellipseList.txt";
#else
  string baseDir = "/Users/vidit/scratch/Data/facesInTheWild/";
  string listFile = "/Users/vidit/scratch/Data/detectionResults/FDDB/imList.txt";
  string detFile = "/Users/vidit/scratch/Data/detectionResults/FDDB/MikolajczykDets.txt";
  string annotFile = "/Users/vidit/scratch/Data/detectionResults/FDDB/ellipseList.txt";
#endif

  // directory containing the images
  string imDir = baseDir;
  // prefix used for writing the ROC-curve files
  string rocFilePrefix = "temp"; 
  // format used for specifying the detected regions
  int detFormat = DET_RECTANGLE;
  // format string to be appended to the image file names in the annotation set
  string annotImageFormat = __IMAGE_FORMAT__;
  std::cout << "annotImageFormat : " << annotImageFormat << std::endl;
  // display the matched pairs
  bool showMatchPairs = false;

  if(argc == 1)
  {
    printUsage();
    return 0;
  }

#ifndef _WIN32
  // parse the input
  while( (c = getopt(argc, argv, "l:r:d:a:z:i:f:s")) != -1){
    switch(c){
      case 'l':
	listFile = optarg;	
	break;
      case 'r':
	rocFilePrefix = optarg;
	break;
      case 'd':
	detFile = optarg;
	break;
      case 'a':
	annotFile = optarg;
	break;
      case 'z':
	annotImageFormat = optarg;
	break;
      case 'i':
	imDir = optarg;
	break;
      case 'f':
	detFormat = atoi(optarg);
    //std::cout << detFormat << " " << optarg << std::endl;
	break;
      case 's':
	showMatchPairs = true;
	break;
      case 'h':
      default:
	printUsage();
	return 0;
    }
  }
#endif

  // read file and compute image-wise scores
  vector<string> *imNames = getImageList(listFile);
  if(imNames == NULL)
  {
    cerr << "No images found in the list " << listFile << endl;	
    cerr << "Set list file using -l option. See usage ./evaluate -h." << endl;
    return -1;

  }

  // file handle for reading annotations
  ifstream fAnnot(annotFile.c_str());
  if(!fAnnot.is_open()){
    cerr << "Can not open annotations from " << annotFile << endl;
    cerr << "Set annotation file using -a option. See usage ./evaluate -h." << endl;
    return -1;
  }

  // file handle for reading detections
  ifstream fDet(detFile.c_str());
  if(!fDet.is_open()){
    cerr << "Can not open detections from " << detFile << endl;
    cerr << "Set detection file using -d option. See usage ./evaluate -h." << endl;
    return -1;
  }

  cout << "Processing " << imNames->size() << " images" << endl;

  // cumRes stores the cumulative results for all the images
  vector<Results *> *cumRes = new vector<Results *>;

  // dummyRes is used for accessing some method from the Results class
  Results *dummyRes = new Results();

  // Process each image
  for(unsigned int ii = 0; ii < imNames->size(); ii++){

    if(ii % 1 == 0)
      cout << ii << " images done" << endl;

    string imName = imNames->at(ii);

    // read image name
    string imS1;
    getline(fAnnot, imS1);

    string imS2;
    getline(fDet, imS2);

    // make sure that the annotation and detections are read for the same image
    if( imName.compare(imS1) || imName.compare(imS2) )
    {
      cerr << imName << " " << imS1 << " " << imS2 << endl;
      cerr << "Incompatible annotation and detection files. See output specifications" << endl;
      return -1; 
    }

    // Read the number of annotations/detections in this image
    int nAnnot, nDet;
    getline(fAnnot, imS1);
    getline(fDet, imS2);

    stringstream ss1(imS1);
    ss1 >> nAnnot;

    stringstream ss2(imS2);
    ss2 >> nDet;

    string imFullName = imDir + imName + annotImageFormat; 
    //std::cout << imFullName << " " << imDir << " " << imName << " " << annotImageFormat << std::endl;

    // Read the annotations
    RegionsSingleImage *annot; 
    annot = new EllipsesSingleImage(imFullName);
    ((EllipsesSingleImage *)annot)->read(fAnnot, nAnnot);

    // Read the detections
    RegionsSingleImage *det;
    switch(detFormat)
    {
      case(DET_ELLIPSE):
	det =  new EllipsesSingleImage(imFullName);
	((EllipsesSingleImage *)det)->read(fDet, nDet);
	break;
      case(DET_RECTANGLE):
	det =  new RectanglesSingleImage(imFullName);
	((RectanglesSingleImage *)det)->read(fDet, nDet);
	break;
      case(DET_PIXELS):
	cerr << "Not yet implemented " << endl;
	assert(false);
      default:
	;
    }

    // imageResults holds the results for different thresholds
    // applied to a single image
    vector<Results *> *imageResults = new vector<Results *>;

    if(nDet == 0)
    {
	// create the image results for zero detections 
	Results *r = new Results(imName, std::numeric_limits<double>::max(), NULL, annot, det);
	imageResults->push_back(r);
    }
    else
    {
      // attach annot and det to the Maching object
      Matching *M = new Matching(annot, det);

      // find the unique values for detection scores
      vector<double> *uniqueScores = det->getUniqueScores();
      sort(uniqueScores->begin(), uniqueScores->end());

      imageResults->reserve(uniqueScores->size());

      // For each unique score value st,
      //  (a) filter the detections with score >= st
      //  (b) compute the matching annot-det pairs
      //  (c) compute the result statistics
      for(vector<double>::iterator uit=uniqueScores->begin(); uit != uniqueScores->end(); ++uit)
      {

	//  (a) filter the detections with score >= st
	double scoreThreshold = *uit;
	for(unsigned di =0; di<det->length(); di++)
	{
	  Region *rd = det->get(di);
	  rd->setValid( rd->detScore >= scoreThreshold );
	}
	  
	//  (b) match annotations to detections
	vector<MatchPair *> *mps = M->getMatchPairs();

	// if the matched pairs are to be displayed
	if(showMatchPairs)
	{
	  if(mps!= NULL)
	  {
	    const IplImage *im = annot->getImage();
	    assert(im != NULL);
	    IplImage *mask = cvCreateImage(cvGetSize(im), im->depth, im->nChannels);
	    cvCopy(im, mask, 0);
	    for(unsigned int i=0; i< mps->size(); i++)
	    {
	      stringstream ss("");
	      ss << i;
	      const char *textDetInd = ss.str().c_str();
	      mask = ( (EllipseR *)(mps->at(i)->r1) )->display(mask, CV_RGB(255,0,0), 3, textDetInd);
	      switch(detFormat)
	      {
		case(DET_RECTANGLE):
		  mask = ( (RectangleR *)(mps->at(i)->r2) )->display(mask, CV_RGB(0,255,0), 3, textDetInd);
		  break;
		case(DET_ELLIPSE):
		  mask = ( (EllipseR *)(mps->at(i)->r2) )->display(mask, CV_RGB(0,255,0), 3, textDetInd);
		  break;
		case(DET_PIXELS):
		  cerr << "Not yet implemented " << endl;
		  return -1;
		  break;
		default:
		  ;
	      }
	    }
	    showImage(" matches ", mask);
	    cvReleaseImage(&mask);
	  }
	}

	//  (c) compute the result statistics and append to the list
	Results *r = new Results(imName, scoreThreshold, mps, annot, det);
	imageResults->push_back(r);

	//r->print(std::cout);
	for(unsigned int mpi=0; mpi < mps->size(); mpi++)
	  delete(mps->at(mpi));
	delete(mps);
      }

      delete(uniqueScores);
      delete(M);
    }
    delete(annot);
    delete(det);

    // merge the list of results for this image (imageResults) with the global list (cumRes)
    vector<Results *> *mergedRes = dummyRes->merge(cumRes, imageResults);

    // free memory for the lists used for merging
    for(unsigned int mi =0; mi<cumRes->size(); mi++)
      delete(cumRes->at(mi));
    delete(cumRes);
    for(unsigned int mi =0; mi<imageResults->size(); mi++)
      delete(imageResults->at(mi));
    delete(imageResults);

    cumRes = mergedRes;
  }

  fAnnot.close();
  fDet.close();

  // save the ROC-curve computed from the cumulative statistics 
  dummyRes->saveROC(rocFilePrefix, cumRes);

  delete(imNames);

  return 0;
}
