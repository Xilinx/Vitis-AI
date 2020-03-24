FACE DETECTION EVALUATION
-------------------------

Author: Vidit Jain < vidit@cs.umass.edu >

This document describes the face detection evaluation toolkit developed as
part of the FDDB benchmark ( http://vis-www.cs.umass.edu/fddb/ ). This
source-code can be used as long as the following reference is cited:

Vidit Jain and Erik Learned-Miller. FDDB: A benchmark for Face Detection in
Unconstrained Settings. Technical Report. University of Massachusetts Amherst. 2010.

Bibtex entry:

@TECHREPORT{fddbTechReport,
  author = {Jain, Vidit and  Learned-Miller, Erik},
  title = {FDDB: A benchmark for Face Detection in Unconstrained Settings.},
  institution = {University of Massachusetts Amherst},
  year = {2010}
}


Compiling the evaluation code (C++):
------------------------------------

1. Requires OpenCV library (http://sourceforge.net/projects/opencvlibrary/)

2. If the utility 'pkg-config' is not available for your operating system,
   edit the Makefile to manually specify the OpenCV flags as following:
   INCS = -I/usr/local/include/opencv
   LIBS = -L/usr/local/lib -lcxcore -lcv -lhighgui -lcvaux -lml

3. The project is compiled by running 'make' in the project directory.

4. Known issue with OpenCV:

  There is an issue with reading some JPG files using cvLoadImage in OpenCV. 
  To avoid this issue in this software, the system utility "convert" is used
  to convert a JPG file to a temporary PPM file, which is then read using OpenCV.

  If you are certain that the OpenCV cvLoadImage works fine on your machine, 
  uncomment line 15 in common.hpp to avoid the expensive conversion to PPM files. 
  Alternatively, convert all the .jpg files to .ppm and modify the common.hpp
  to use ".ppm" for the __IMAGE_FORMAT__


Usage for the evaluation binary:
------------------------------

./evaluate [OPTIONS]
   -h              : print usage
   -a fileName     : file with face annotations (default: ellipseList.txt)
   -d fileName     : file with detections (default: faceList.txt)
   -f format       : representation of faces in the detection file (default: 0)
                     [ 0 (rectangle), 1 (ellipse), or  2 (pixels) ]
   -i dirName      : directory where the original images are stored 
		     (default: ~/scratch/Data/facesInTheWild/)
   -l fileName     : file with list of images to be evaluated (default: temp.txt)
   -r fileName     : prefix for files to store the ROC curves (default: temp)
   -s              : display the matching detection-annotation pairs.
   -z imageFormat  : image format used for reading images for the annotation set 
                     (default: .jpg )


Perl wrapper for FDDB evaluation:
---------------------------------
Requires: GNUPLOT

The wrapper 'runEvaluate.pl' executes the following in a sequence:

1. reads the detection output for different folds from a directory
2. reads the annotations from a file
3. performs the evaluation
4. use GNUplot to create the discrete and continuous versions of ROC curves.

Edit the Perl code to modify the following variables:
1. $GNUPLOT     -- path to gnuplot
2. $evaluateBin -- the evaluation binary compiled as above
3. $imDir       -- where the image files are stored  
4. $fddbDir     -- where the FDDB folds are stored
5. $detDir      -- the  directory containing detection results for different folds. In this
                   directory, the outputs for different folds are expected to be in files 
                   'fold-%02d-out.txt'
6. $detFormat   -- the specification for the detection output (See -f option for the evaluation binary)
