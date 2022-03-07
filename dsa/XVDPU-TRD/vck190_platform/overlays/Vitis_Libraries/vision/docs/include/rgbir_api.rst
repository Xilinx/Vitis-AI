.. _rgbirbayer:

RGBIR  to Standard Bayer Format
================================

The ``rgbir2bayer`` function creates a standard RGB-only-mosaic
and an IR image from input RGB-IR combined mosaic image. 

|rgbir2bayer|

.. rubric:: API Syntax

.. code:: c

	template <int FSIZE1 = 5, int FSIZE2 = 3, int BFORMAT = 0,
          int TYPE, int ROWS, int COLS, int NPPC = 1,
          int XFCV_DEPTH, int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0>
	void rgbir2bayer(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _src,
                    char R_IR_C1_wgts[FSIZE1 * FSIZE1],
                    char R_IR_C2_wgts[FSIZE1 * FSIZE1],
                    char B_at_R_wgts[FSIZE1 * FSIZE1],
                    char IR_at_R_wgts[FSIZE2 * FSIZE2],
                    char IR_at_B_wgts[FSIZE2 * FSIZE2],
                    char sub_wgts[4],
                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst_rggb,
                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst_ir)

.. rubric:: Parameter Descriptions


The following table describes the template and the function parameters.

.. table:: Table . rgbir2bayer Parameter Description

   +-------------------+--------------------------------------------------+
   | Parameter         | Description                                      |
   +===================+==================================================+
   | FSIZE1            | Filter size for RGB pixels                       |
   +-------------------+--------------------------------------------------+
   | FSIZE2            | Filter size for IR pixels                        |
   +-------------------+--------------------------------------------------+
   | BFORMAT           | Bayer format. Supported types are XF_BAYER_GR    |
   |                   | and XF_BAYER_BG                                  |
   +-------------------+--------------------------------------------------+
   | TYPE              | Input pixel Type. 8-bit, 10 bit, 12 bit and 16   |
   |                   | bit unsigned, 1                                  |
   |                   | channel is supported                             |
   |                   | (XF_8UC1,XF_10UC1, XF_12UC1, XF_16UC1).          |
   +-------------------+--------------------------------------------------+
   | ROWS              | Maximum height of input and output image.        |
   +-------------------+--------------------------------------------------+
   | COLS              | Maximum width of input and output image. Must be |
   |                   | multiple of NPC.                                 |
   +-------------------+--------------------------------------------------+
   | NPPC              | Number of pixels to be processed per cycle,      |
   |                   | possible options are XF_NPPC1 only.              |
   +-------------------+--------------------------------------------------+
   | XFCV_DEPTH        | Depth of the hls::stream formed by the xf::Mat   |
   +-------------------+--------------------------------------------------+
   | BORDER_T          | Border handling type. Fixed to XF_BORDER_CONSTANT|
   +-------------------+--------------------------------------------------+
   | USE_URAM          | Enable URAM storage strucure                     |
   +-------------------+--------------------------------------------------+
   | \_src_mat         | Input image                                      |
   +-------------------+--------------------------------------------------+
   | R_IR_C1_wgts      | Weights to calculate R at IR location for        |
   |                   | constellation 1                                  |
   +-------------------+--------------------------------------------------+
   | R_IR_C2_wgts      | Weights to calculate R at IR location for        |
   |                   | constellation 2                                  |
   +-------------------+--------------------------------------------------+
   | B_at_R_wgts       | Weights to calculate B at R location             |
   +-------------------+--------------------------------------------------+
   | IR_at_R_wgts      | Weights to calculate IR at R location            |
   +-------------------+--------------------------------------------------+
   | IR_at_B_wgts      | Weights to calculate IR at B location            |
   +-------------------+--------------------------------------------------+
   | sub_wgts          | Weights to perform weighted subtraction of IR    |
   |                   | image from RGB image. sub_wgts[0] -> G Pixel,    |
   |                   | sub_wgts[1] -> R Pixel, sub_wgts[2] -> B Pixel   |
   |                   | sub_wgts[3] -> calculated B Pixel                |
   +-------------------+--------------------------------------------------+
   | _dst_rggb         | output image in standard bayer format with only  |
   |                   | R,G,B pixels                                     |
   +-------------------+--------------------------------------------------+
   | _dst_ir           | IR output image with only IR pixels              |
   +-------------------+--------------------------------------------------+
   
.. rubric:: Resource Utilization


The following table summarizes the resource utilization in different configurations, generated using Vitis HLS 2021.1 tool for the
Xczu9eg-ffvb1156-1-i-es1 FPGA.

.. table:: Table . rgbir2bayer Function Resource Utilization Summary

    +----------------+---------------------------+----------------------+-----------+------+------+-----+
    | Operating Mode | Operating Frequency (MHz) |               Utilization Estimate                   |
    +                +                           +----------------------+-----------+------+------+-----+
    |                |                           | BRAM_18K             | DSP_48Es  |  FF  |  LUT | CLB |
    +================+===========================+======================+===========+======+======+=====+
    | 1 Pixel        | 300                       |     37               |    0      | 4345 | 6243 |1366 |
    +----------------+---------------------------+----------------------+-----------+------+------+-----+


.. rubric:: Performance Estimate


The following table summarizes the performance of the kernel in 1-pixel
mode as generated using Vitis HLS 2021.1 tool for the Xilinx
xczu9eg-ffvb1156-2-i-es2 FPGA to process a grayscale 4K (2160x3840)
image.

.. table:: Table . rgbir2bayer Function Performance Estimate Summary

    +-----------------------------+------------------+
    | Operating Mode              | Latency Estimate |
    +                             +------------------+
    |                             | Max Latency (ms) |
    +=============================+==================+
    | 1 pixel operation (300 MHz) |       27.7       |
    +-----------------------------+------------------+
	
.. |rgbir2bayer| image:: ./images/rgbir2bayer.PNG
   :class: image