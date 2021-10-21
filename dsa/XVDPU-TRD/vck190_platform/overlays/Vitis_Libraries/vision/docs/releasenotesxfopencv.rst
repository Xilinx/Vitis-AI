
.. meta::
   :keywords: New features
   :description: Release notes.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _releasenotes-xfopencv:


Release notes
##############

The below section explains the new features added and also changes in the existing library along with the known issues.

-  `New features and functions <#pl-new>`_
-  `Library related changes <#library-changes>`_
-  `Known issues <#known-issues>`_


New features and functions
============================

The below functions and pipelines are newly added into the library.

* ISP pipeline and functions


	* Updated 2020.2 Non-HDR Pipeline : 

		* Support to change few of the ISP parameters at runtime : Gain parameters for red and blue channels, AWB enable/disable option, gamma tables for R,G,B, %pixels ignored to compute min&max for awb normalization.
		* Gamma Correction and Color Space conversion (RGB2YUYV) made part of the pipeline.
		
	* New 2021.1 HDR Pipeline : 2020.2 Pipeline + HDR support

		 * HDR merge : merges the 2 exposures which supports sensors with digital overlap between short exposure frame and long exposure frame.
		 * Four Bayer patterns supported : RGGB,BGGR,GRBG,GBRG
		 * HDR merge + isp pipeline with runtime configurations, which returns RGB output.
		 * Extraction function : HDR extraction function is preprocessing function, which takes single digital overlapped stream as input and returns the 2 output exposure frames(SEF,LEF). 
		 
* 3DLUT

	3DLUT provides input-output mapping to control complex color operators, such as hue, saturation, and luminance.
 
* CLAHE 


	Contrast Limited Adaptive Histogram Equalization is a method which limits the contrast while performing adaptive histogram equalization so that it does not over amplify the contrast in the near constant regions. This it also reduces the problem of noise amplification.

* Flip 


	Flips the image along horizontal and vertical line.

* Custom CCA 


	Custom version of Connected Component Analysis Algorithm for defect detection in fruits. Apart from computing defected portion of fruits , it computes defected-pixels as well as total-fruit-pixels

* Other updates


	* Canny updates : Canny function now supports any image resolution.
	* Gamma correction : Gamma function changed
	* AWB optimization : AWB modules optimized.


.. _library-changes:

Library related changes
=======================

* All tests have been upgraded from using OpenCV 3.4.2 to OpenCV 4.4.0
* Added support for vck190 and aws-vu9p-f1 platforms.
* A new benchmarking section with benchmarking collateral for selected pipeline/functions published.

.. _known-issues:

Known issues
============

* Vitis GUI projects on RHEL83 and CEntOS82 may fail because of a lib conflict in the LD_LIBRARY_PATH setting.
* Windows OS has path length limitations, kernel names must be smaller than 25 characters.





















