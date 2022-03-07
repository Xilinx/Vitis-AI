
.. meta::
   :keywords: Vision, Library, Vitis Vision Library, cv
   :description: Using the Vitis vision library.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


Using the Vitis vision Library
===============================

This section describes using the Vitis vision library in the Vitis development
environment.

Note: The instructions in this section assume that you have downloaded
and installed all the required packages. 

*include* folder constitutes all the necessary components to build a
Computer Vision or Image Processing pipeline using the library. The
folders *common* and *core* contain the infrastructure that the library
functions need for basic functions, Mat class, and macros. The library
functions are categorized into 4 folders, *features*, *video*, *dnn*, and
*imgproc* based on the operation they perform. The names of the folders
are self-explanatory.

To work with the library functions, you need to include the path to the
the *include* folder in the Vitis project. You can include relevant header files
for the library functions you will be working with after you source the
*include* folder’s path to the compiler. For example, if you would like to
work with Harris Corner Detector and Bilateral Filter, you must use the
following lines in the host code:

.. code:: c

   #include “features/xf_harris.hpp” //for Harris Corner Detector
   #include “imgproc/xf_bilateral_filter.hpp” //for Bilateral Filter
   #include “video/xf_kalmanfilter.hpp”

After the headers are included, you can work with the library functions
as described in the `Vitis vision Library API
Reference <api-reference.html#ycb1504034263746>`__ using the examples
in the examples folder as reference.

The following table gives the name of the header file, including the
folder name, which contains the library function.

.. table:: Table : Vitis Vision Library 

   +-------------------------------------------+-----------------------------------+
   | Function Name                             | File Path in the include folder   |
   +===========================================+===================================+
   | xf::cv::accumulate                        | imgproc/xf_accumulate_image.hpp   |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::accumulateSquare                  | imgproc/xf_accumulate_squared.hpp |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::accumulateWeighted                | imgproc/xf_accumulate_weighted.hp |
   |                                           | p                                 |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::absdiff, xf::cv::add,             | core/xf_arithm.hpp                |
   | xf::cv::subtract, xf::cv::bitwise_and,    |                                   |
   | xf::cv::bitwise_or, xf::cv::bitwise_not,  |                                   |
   | xf::cv::bitwise_xor,xf::cv::multiply      |                                   |
   | ,xf::cv::Max, xf::cv::Min,xf::cv::compare,|                                   |
   | xf::cv::zero, xf::cv::addS, xf::cv::SubS, |                                   |
   | xf::cv::SubRS ,xf::cv::compareS,          |                                   |
   | xf::cv::MaxS, xf::cv::MinS, xf::cv::set   |                                   |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::addWeighted                       | imgproc/xf_add_weighted.hpp       |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::autowhitebalance                  | imgproc/xf_autowhitebalance.hpp   |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::autoexposurecorrection            | imgproc/xf_aec.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::bilateralFilter                   | imgproc/xf_bilaterealfilter.hpp   |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::blackLevelCorrection              | imgproc/xf_black_level.hpp        |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::bfmatcher                         | imgproc/xf_bfmatcher.hpp          |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::boxFilter                         | imgproc/xf_box_filter.hpp         |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::boundingbox                       | imgproc/xf_boundingbox.hpp        |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::badpixelcorrection                | imgproc/xf_bpc.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Canny                             | imgproc/xf_canny.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::colorcorrectionmatrix             | imgproc/xf_colorcorrectionmatrix. |
   |                                           | hpp                               |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Colordetect                       | imgproc/xf_colorthresholding.hpp, |
   |                                           | imgproc/xf_bgr2hsv.hpp,           |
   |                                           | imgproc/xf_erosion.hpp,           |
   |                                           | imgproc/xf_dilation.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::merge                             | imgproc/xf_channel_combine.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::extractChannel                    | imgproc/xf_channel_extract.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::ccaCustom                         | imgproc/xf_cca_custom.hpp         |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::clahe                             | imgproc/xf_clahe.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::convertTo                         | imgproc/xf_convert_bitdepth.hpp   |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::crop                              | imgproc/xf_crop.hpp               |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::distanceTransform                 | imgproc/xf_distancetransform.hpp  |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::nv122iyuv, xf::cv::nv122rgba,     | imgproc/xf_cvt_color.hpp          |
   | xf::cv::nv122yuv4, xf::cv::nv212iyuv,     |                                   |
   | xf::cv::nv212rgba, xf::cv::nv212yuv4,     |                                   |
   | xf::cv::rgba2yuv4, xf::cv::rgba2iyuv,     |                                   |
   | xf::cv::rgba2nv12, xf::cv::rgba2nv21,     |                                   |
   | xf::cv::uyvy2iyuv, xf::cv::uyvy2nv12,     |                                   |
   | xf::cv::uyvy2rgba, xf::cv::yuyv2iyuv,     |                                   |
   | xf::cv::yuyv2nv12, xf::cv::yuyv2rgba,     |                                   |
   | xf::cv::rgb2iyuv,xf::cv::rgb2nv12,        |                                   |
   | xf::cv::rgb2nv21, xf::cv::rgb2yuv4,       |                                   |
   | xf::cv::rgb2uyvy, xf::cv::rgb2yuyv,       |                                   |
   | xf::cv::rgb2bgr, xf::cv::bgr2uyvy,        |                                   |
   | xf::cv::bgr2yuyv, xf::cv::bgr2rgb,        |                                   |
   | xf::cv::bgr2nv12, xf::cv::bgr2nv21,       |                                   |
   | xf::cv::iyuv2nv12, xf::cv::iyuv2rgba,     |                                   |
   | xf::cv::iyuv2rgb, xf::cv::iyuv2yuv4,      |                                   |
   | xf::cv::nv122uyvy, xf::cv::nv122yuyv,     |                                   |
   | xf::cv::nv122nv21, xf::cv::nv212rgb,      |                                   |
   | xf::cv::nv212bgr, xf::cv::nv212uyvy,      |                                   |
   | xf::cv::nv212yuyv, xf::cv::nv212nv12,     |                                   |
   | xf::cv::uyvy2rgb, xf::cv::uyvy2bgr,       |                                   |
   | xf::cv::uyvy2yuyv, xf::cv::yuyv2rgb,      |                                   |
   | xf::cv::yuyv2bgr, xf::cv::yuyv2uyvy,      |                                   |
   | xf::cv::rgb2gray, xf::cv::bgr2gray,       |                                   |
   | xf::cv::gray2rgb, xf::cv::gray2bgr,       |                                   |
   | xf::cv::rgb2xyz, xf::cv::bgr2xyz...       |                                   |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::densePyrOpticalFlow               | video/xf_pyr_dense_optical_flow.h |
   |                                           | pp                                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::DenseNonPyrLKOpticalFlow          | video/xf_dense_npyr_optical_flow. |
   |                                           | hpp                               |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::dilate                            | imgproc/xf_dilation.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::demosaicing                       | imgproc/xf_demosaicing.hpp        |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::erode                             | imgproc/xf_erosion.hpp            |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::fast                              | features/xf_fast.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::filter2D                          | imgproc/xf_custom_convolution.hpp |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::flip                              | features/xf_flip.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::GaussianBlur                      | imgproc/xf_gaussian_filter.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::gaincontrol                       | imgproc/xf_gaincontrol.hpp        |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::gammacorrection                   | imgproc/xf_gammacorrection.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::gtm                               | imgproc/xf_gtm.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::cornerHarris                      | features/xf_harris.hpp            |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::calcHist                          | imgproc/xf_histogram.hpp          |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::equalizeHist                      | imgproc/xf_hist_equalize.hpp      |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::extractExposureFrames             | imgproc/xf_extract_eframes.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::HDRMerge_bayer                    | imgproc/xf_hdrmerge.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::HOGDescriptor                     | imgproc/xf_hog_descriptor.hpp     |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Houghlines                        | imgproc/xf_houghlines.hpp         |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::inRange                           | imgproc/xf_inrange.hpp            |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::integralImage                     | imgproc/xf_integral_image.hpp     |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::KalmanFilter                      | video/xf_kalmanfilter.hpp         |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Lscdistancebased                  | imgproc/xf_lensshadingcorrection  |
   |                                           | .hpp                              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::LTM::process                      | imgproc/xf_ltm.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::LUT                               | imgproc/xf_lut.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::magnitude                         | core/xf_magnitude.hpp             |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::MeanShift                         | imgproc/xf_mean_shift.hpp         |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::meanStdDev                        | core/xf_mean_stddev.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::medianBlur                        | imgproc/xf_median_blur.hpp        |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::minMaxLoc                         | core/xf_min_max_loc.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::modefilter                        | imgproc/xf_modefilter.hpp         |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::OtsuThreshold                     | imgproc/xf_otsuthreshold.hpp      |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::phase                             | core/xf_phase.hpp                 |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::preProcess                        | dnn/xf_pre_process.hpp            |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::paintmask                         | imgproc/xf_paintmask.hpp          |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::pyrDown                           | imgproc/xf_pyr_down.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::pyrUp                             | imgproc/xf_pyr_up.hpp             |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::xf_QuatizationDithering           | imgproc/xf_quantizationdithering  |
   |                                           | .hpp                              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::reduce                            | imgrpoc/xf_reduce.hpp             |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::remap                             | imgproc/xf_remap.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::resize                            | imgproc/xf_resize.hpp             |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::rgbir2bayer                       | imgproc/xf_rgbir.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::convertScaleAbs                   | imgproc/xf_convertscaleabs.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Scharr                            | imgproc/xf_scharr.hpp             |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::SemiGlobalBM                      | imgproc/xf_sgbm.hpp               |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Sobel                             | imgproc/xf_sobel.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::StereoPipeline                    | imgproc/xf_stereo_pipeline.hpp    |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::sum                               | imgproc/xf_sum.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::StereoBM                          | imgproc/xf_stereoBM.hpp           |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::SVM                               | imgproc/xf_svm.hpp                |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::lut3d                             | imgproc/xf_3dlut.hpp              |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::Threshold                         | imgproc/xf_threshold.hpp          |
   +-------------------------------------------+-----------------------------------+
   | xf::cv::warpTransform                     | imgproc/xf_warp_transform.hpp     |
   +-------------------------------------------+-----------------------------------+




Changing the Hardware Kernel Configuration
------------------------------------------

   To modify the configuration of any function, update the following file:
   
   <path to vitis vision git folder>/vision/L1/examples/<function>/build/xf_config_params.h .


Using the Vitis vision Library Functions on Hardware
----------------------------------------------------

The following table lists the Vitis vision library functions and the command
to run the respective examples on hardware. It is assumed that your
design is completely built and the board has booted up correctly.

.. table:: Table : Using the Vitis vision Library Function on Hardware

   +--------------+---------------------------+--------------------------+
   | Example      | Function Name             | Usage on Hardware        |
   +==============+===========================+==========================+
   | accumulate   | xf::cv::accumulate        | ./<executable name>.elf  |
   |              |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | accumulatesq | xf::cv::accumulateSquare  | ./<executable name>.elf  |
   | uared        |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | accumulatewe |xf::cv::accumulateWeighted | ./<executable name>.elf  |
   | ighted       |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | addS         | xf::cv::addS              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | arithm       | xf::cv::absdiff, 	      | ./<executable name>.elf  |
   |              | xf::cv::subtract,         | <path to input image 1>  |
   |              | xf::cv::bitwise_and,      | <path to input image 2>  |
   |              | xf::cv::bitwise_or,       |                          |
   |              | xf::cv::bitwise_not,      |                          |
   |              | xf::cv::bitwise_xor       |                          |
   +--------------+---------------------------+--------------------------+
   | addweighted  | xf::cv::addWeighted       | ./<executable name>.elf  |
   |              |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | Autoexposure | xf::cv::autoexposurecorr  | ./<executable name>.elf  |
   | correction   | ection                    | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Autowhite    | xf::cv::autowhitebalance  | ./<executable name>.elf  |
   | balance      |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Bilateralfil | xf::cv::bilateralFilter   | ./<executable name>.elf  |
   | ter          |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | BlackLevel   | xf::cv::blackLevel        | ./<executable name>.elf  |
   | Correction   | Correction                | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | BruteForce   | xf::cv::bfmatcher         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |  
   +--------------+---------------------------+--------------------------+   
   | Boxfilter    | xf::cv::boxFilter         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Badpixelcorr | xf::cv::badpixelcorrection| ./<executable name>.elf  |
   | ection       |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Boundingbox  | xf::cv::boundingbox       | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   |              |                           | <No of ROI's>            |
   +--------------+---------------------------+--------------------------+
   | Canny        | xf::cv::Canny             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | ccaCustom    | xf::cv::ccaCustom         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | channelcombi | xf::cv::merge             | ./<executable name>.elf  |
   | ne           |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   |              |                           | <path to input image 3>  |
   |              |                           | <path to input image 4>  |
   +--------------+---------------------------+--------------------------+
   | Channelextra | xf::cv::extractChannel    | ./<executable name>.elf  |
   | ct           |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | CLAHE        | xf::cv::clahe             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Colordetect  | xf::cv::bgr2hsv,          | ./<executable name>.elf  |
   |              | xf::cv::colorthresholding,| <path to input image>    |
   |              | xf::cv:: erode, xf::cv::  |                          |
   |              | dilate                    |                          |
   +--------------+---------------------------+--------------------------+
   | color        | xf::cv::colorcorrection   | ./<executable name>.elf  |
   | correction   | matrix                    | <path to input image>    |
   | matrix       |                           |                          |
   +--------------+---------------------------+--------------------------+
   | compare      | xf::cv::compare           | ./<executable name>.elf  |
   |              |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | compareS     | xf::cv::compareS          | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Convertbitde | xf::cv::convertTo         | ./<executable name>.elf  |
   | pth          |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | convertScale | xf::cv::convertScaleAbs   | ./<executable name>.elf  |
   | Abs          |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Cornertracke | xf::cv::cornerTracker     | ./exe <input video> <no. |
   | r            |                           | of frames> <Harris       |
   |              |                           | Threshold> <No. of       |
   |              |                           | frames after which       |
   |              |                           | Harris Corners are       |
   |              |                           | Reset>                   |
   +--------------+---------------------------+--------------------------+
   | crop         | xf::cv::crop              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Customconv   | xf::cv::filter2D          | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::iyuv2nv12         | ./<executable name>.elf  |
   | IYUV2NV12    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   |              |                           | <path to input image 3>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::iyuv2rgba         | ./<executable name>.elf  |
   | IYUV2RGBA    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   |              |                           | <path to input image 3>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::iyuv2yuv4         | ./<executable name>.elf  |
   | IYUV2YUV4    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   |              |                           | <path to input image 3>  |
   |              |                           | <path to input image 4>  |
   |              |                           | <path to input image 5>  |
   |              |                           | <path to input image 6>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::nv122iyuv         | ./<executable name>.elf  |
   | NV122IYUV    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::nv122rgba         | ./<executable name>.elf  |
   | NV122RGBA    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::nv122yuv4         | ./<executable name>.elf  |
   | NV122YUV4    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::nv212iyuv         | ./<executable name>.elf  |
   | NV212IYUV    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::nv212rgba         | ./<executable name>.elf  |
   | NV212RGBA    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::nv212yuv4         | ./<executable name>.elf  |
   | NV212YUV4    |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::rgba2yuv4         | ./<executable name>.elf  |
   | RGBA2YUV4    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::rgba2iyuv         | ./<executable name>.elf  |
   | RGBA2IYUV    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::rgba2nv12         | ./<executable name>.elf  |
   | RGBA2NV12    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::rgba2nv21         | ./<executable name>.elf  |
   | RGBA2NV21    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::uyvy2iyuv         | ./<executable name>.elf  |
   | UYVY2IYUV    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::uyvy2nv12         | ./<executable name>.elf  |
   | UYVY2NV12    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::uyvy2rgba         | ./<executable name>.elf  |
   | UYVY2RGBA    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::yuyv2iyuv         | ./<executable name>.elf  |
   | YUYV2IYUV    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::yuyv2nv12         | ./<executable name>.elf  |
   | YUYV2NV12    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | cvtcolor     | xf::cv::yuyv2rgba         | ./<executable name>.elf  |
   | YUYV2RGBA    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Demosaicing  | xf::cv::demosaicing       | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Difference   | xf::cv::GaussianBlur,     | ./<exe-name>.elf <path   |
   | of Gaussian  | xf::cv::duplicateMat,     | to input image>          |
   |              | and                       |                          |
   |              | xf::cv::subtract          |                          |
   +--------------+---------------------------+--------------------------+
   | Dilation     | xf::cv::dilate            | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Distance     | xf::cv::distanceTransform | ./<executable name>.elf  |
   | Transform    |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Erosion      | xf::cv::erode             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | FAST         | xf::cv::fast              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Flip         | xf::cv::flip              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Gaussianfilt | xf::cv::GaussianBlur      | ./<executable name>.elf  |
   | er           |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Gaincontrol  | xf::cv::gaincontrol       | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Gammacorrec  | xf::cv::gammacorrection   | ./<executable name>.elf  |
   | tion         |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Global Tone  | xf::cv::gtm               | ./<executable name>.elf  |
   | Mapping      |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Harris       | xf::cv::cornerHarris      | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Histogram    | xf::cv::calcHist          | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Histequializ | xf::cv::equalizeHist      | ./<executable name>.elf  |
   | e            |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Hog          | xf::cv::HOGDescriptor     | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Houghlines   | xf::cv::HoughLines        | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | inRange      | xf::cv::inRange           | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Integralimg  | xf::cv::integralImage     | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Laplacian    | xf::cv::filter2d          | ./<executable name>.elf  |
   | Filter       |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Lkdensepyrof | xf::cv::densePyrOpticalFlo| ./<executable name>.elf  |
   |              | w                         | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | Lknpyroflow  | xf::cv::DenseNonPyr       | ./<executable name>.elf  |
   |              | LKOpticalFlow             | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | lensshading  | xf::cv::Lscdistancebased  | ./<executable name>.elf  |
   | correction   |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Lut          | xf::cv::LUT               | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Local tone   | xf::cv::LTM::process      | ./<executable name>.elf  |
   | mapping      |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Kalman       | xf::cv::KalmanFilter      | ./<executable name>.elf  |
   | Filter       |                           |                          |
   +--------------+---------------------------+--------------------------+
   | Magnitude    | xf::cv::magnitude         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Max          | xf::cv::Max               | ./<executable name>.elf  |
   |              |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | MaxS         | xf::cv::MaxS              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | meanshifttra | xf::cv::MeanShift         | ./<executable name>.elf  |
   | cking        |                           | <path to input           |
   |              |                           | video/input image files> |
   |              |                           | <Number of objects to    |
   |              |                           | track>                   |
   +--------------+---------------------------+--------------------------+
   | meanstddev   | xf::cv::meanStdDev        | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | medianblur   | xf::cv::medianBlur        | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Min          | xf::cv::Min               | ./<executable name>.elf  |
   |              |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | MinS         | xf::cv::MinS              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Minmaxloc    | xf::cv::minMaxLoc         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Mode filter  | xf::cv::modefilter        | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | otsuthreshol | xf::cv::OtsuThreshold     | ./<executable name>.elf  |
   | d            |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | paintmask    | xf::cv::paintmask         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Phase        | xf::cv::phase             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Pyrdown      | xf::cv::pyrDown           | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Pyrup        | xf::cv::pyrUp             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | Quantization | xf::cv::xf_Quatization    | ./<executable name>.elf  |
   | Dithering    | Dithering                 | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | reduce       | xf::cv::reduce            | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | remap        | xf::cv::remap             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   |              |                           | <path to mapx data>      |
   |              |                           | <path to mapy data>      |
   +--------------+---------------------------+--------------------------+
   | Resize       | xf::cv::resize            | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | rgbir2bayer  | xf::cv::rgbir2bayer       | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | scharrfilter | xf::cv::Scharr            | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | set          | xf::cv::set               | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | SemiGlobalBM | xf::cv::SemiGlobalBM      | ./<executable name>.elf  |
   |              |                           | <path to left image>     |
   |              |                           | <path to right image>    |
   +--------------+---------------------------+--------------------------+
   | sobelfilter  | xf::cv::Sobel             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | stereopipeli | xf::cv::StereoPipeline    | ./<executable name>.elf  |
   | ne           |                           | <path to left image>     |
   |              |                           | <path to right image>    |
   +--------------+---------------------------+--------------------------+
   | stereolbm    | xf::cv::StereoBM          | ./<executable name>.elf  |
   |              |                           | <path to left image>     |
   |              |                           | <path to right image>    |
   +--------------+---------------------------+--------------------------+
   | subRS        | xf::cv::SubRS             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | subS         | xf::cv::SubS              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | sum          | xf::cv::sum               | ./<executable name>.elf  |
   |              |                           | <path to input image 1>  |
   |              |                           | <path to input image 2>  |
   +--------------+---------------------------+--------------------------+
   | Svm          | xf::cv::SVM               | ./<executable name>.elf  |
   +--------------+---------------------------+--------------------------+
   | threshold    | xf::cv::Threshold         | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | 3dlut        | xf::cv::lut3d             | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | warptransfor | xf::cv::warpTransform     | ./<executable name>.elf  |
   | m            |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+
   | zero         | xf::cv::zero              | ./<executable name>.elf  |
   |              |                           | <path to input image>    |
   +--------------+---------------------------+--------------------------+

   