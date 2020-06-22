/*
Apache License

Version 2.0, January 2004

http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.

"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:

You must give any other recipients of the Work or Derivative Works a copy of this License; and
You must cause any modified files to carry prominent notices stating that You changed the files; and
You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and
If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License.

You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.
5. Submission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS
*/
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
