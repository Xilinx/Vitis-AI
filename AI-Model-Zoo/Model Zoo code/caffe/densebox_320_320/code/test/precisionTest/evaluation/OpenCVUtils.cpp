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
