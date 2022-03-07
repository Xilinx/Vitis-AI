
.. meta::
   :keywords: Vision, Library, Vitis Vision Library, design, methodology, OpenCL, OpenCV, libOpenCL
   :description: Describes the methodology to create a kernel, corresponding host code and a suitable makefile to compile an Vitis Vision kernel for any of the supported platforms in Vitis.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

Getting Started with Vitis Vision
#################################

Describes the methodology to create a kernel, corresponding host code and a suitable
makefile to compile an Vitis Vision kernel for any of the supported
platforms in Vitis. The subsequent section also explains the
methodology to verify the kernel in various emulation modes and on the
hardware.

.. _prerequisites_hls:

Prerequisites
=============

#. Valid installation of Vitis™ 2021.2 or later version and the
   corresponding licenses.
#. Install the Vitis Vision libraries, if you intend to use libraries
   compiled differently than what is provided in Vitis.
#. Install the card for which the platform is supported in Vitis 2021.2 or
   later versions.
#. If targeting an embedded platform, set up the evaluation board.
#. Xilinx® Runtime (XRT) must be installed. XRT provides software
   interface to Xilinx FPGAs.
#. Install/compile OpenCV libraries(with compatible libjpeg.so). 
   Appropriate version (X86/aarch32/aarch64) of compiler must be used based 
   on the available processor for the target board.
#. libOpenCL.so must be installed if not present along with the
   platform.

.. note:: All Vitis Vision functions were tested against OpenCV version - 4.4.0

Vitis Design Methodology
=========================

There are three critical components in making a kernel work on a
platform using Vitis™:

#. Host code with OpenCL constructs
#. Wrappers around HLS Kernel(s)
#. Makefile to compile the kernel for emulation or running on hardware.


Host Code with OpenCL
---------------------

Host code is compiled for the host machine that runs on the host and
provides the data and control signals to the attached hardware with the
FPGA. The host code is written using OpenCL constructs and provides
capabilities for setting up, and running a kernel on the FPGA. The
following functions are executed using the host code:

#. Loading the kernel binary on the FPGA – xcl::import_binary_file()
   loads the bitstream and programs the FPGA to enable required
   processing of data.
#. Setting up memory buffers for data transfer – Data needs to be sent
   and read from the DDR memory on the hardware. cl::Buffers are created
   to allocate required memory for transferring data to and from the
   hardware.
#. Transfer data to and from the hardware –enqueueWriteBuffer() and
   enqueueReadBuffer() are used to transfer the data to and from the
   hardware at the required time.
#. Execute kernel on the FPGA – There are functions to execute kernels
   on the FPGA. There can be single kernel execution or multiple kernel
   execution that could be asynchronous or synchronous with each other.
   Commonly used command is enqueueTask().
#. Profiling the performance of kernel execution – The host code in
   OpenCL also enables measurement of the execution time of a kernel on
   the FPGA. The function used in our examples for profiling is
   getProfilingInfo().


Wrappers around HLS Kernel(s)
-----------------------------

All Vitis Vision kernels are provided with C++ function templates (located
at <Github repo>/include) with image containers as objects of xf::cv::Mat
class. In addition, these kernels will work either in stream based
(where complete image is read continuously) or memory mapped (where
image data access is in blocks).

Vitis flow (OpenCL) requires kernel interfaces to be memory pointers
with width in power(s) of 2. So glue logic is required for converting
memory pointers to xf::cv::Mat class data type and vice-versa when
interacting with Vitis Vision kernel(s). Wrapper(s) are build over the
kernel(s) with this glue logic. Below examples will provide a
methodology to handle different kernel (Vitis Vision kernels located at
<Github repo>/include) types (stream and memory mapped).


Stream Based Kernels
~~~~~~~~~~~~~~~~~~~~

To facilitate the conversion of pointer to xf::Mat and vice versa, two
adapter functions are included as part of Vitis Vision xf::cv::Array2xfMat() and
xf::cv::xfMat2Array(). It is necessary for the xf::Mat objects to be invoked
as streams using HLS pragma with a minimum depth of 2. This results in a
top-level (or wrapper) function for the kernel as shown below:

.. code:: c

   extern “C” 
   { 
   void func_top (ap_uint *gmem_in, ap_uint *gmem_out, ...) { 
   xf::cv::Mat<…> in_mat(…), out_mat(…);
   #pragma HLS dataflow 
   xf::cv::Array2xfMat<…> (gmem_in, in_mat); 
   xf::cv::Vitis Vision-func<…> (in_mat, out_mat…); 
   xf::cv::xfMat2Array<…> (gmem_out, out_mat); 
   }
   }

The above illustration assumes that the data in xf::cv::Mat is being
streamed in and streamed out. You can also create a pipeline with
multiple functions in pipeline instead of just one Vitis Vision function.

For the stream based kernels with different inputs of different sizes,
multiple instances of the adapter functions are necessary. For this,

.. code:: c

   extern “C” { 
   void func_top (ap_uint *gmem_in1, ap_uint *gmem_in2, ap_uint *gmem_in3, ap_uint *gmem_out, ...) { 
   xf::cv::Mat<...,HEIGHT,WIDTH,…> in_mat1(…), out_mat(…);
   xf::cv::Mat<...,HEIGHT/4,WIDTH,…>  in_mat2(…), in_mat3(…); 
   #pragma HLS dataflow 
   xf::cv::accel_utils obj_a, obj_b;
   obj_a.Array2xfMat<…,HEIGHT,WIDTH,…> (gmem_in1, in_mat1);
   obj_b.Array2xfMat<…,HEIGHT/4,WIDTH,…> (gmem_in2, in_mat2); 
   obj_b.Array2xfMat<…,HEIGHT/4,WIDTH,…> (gmem_in3, in_mat3); 
   xf::cv::Vitis-Vision-func(in_mat1, in_mat2, int_mat3, out_mat…); 
   xf::cv::xfMat2Array<…> (gmem_out, out_mat); 
   }
   }

For the stream based implementations, the data must be fetched from the
input AXI and must be pushed to xfMat as required by the xfcv kernels
for that particular configuration. Likewise, the same operations must be
performed for the output of the xfcv kernel. To perform this, two
utility functions are provided, xf::cv::Array2xfMat() and xf::cv::xfMat2Array().

Array2xfMat
~~~~~~~~~~~

This function converts the input array to xf::cv::Mat. The Vitis Vision kernel
would require the input to be of type, xf::cv::Mat. This function would read
from the array pointer and write into xf::cv::Mat based on the particular
configuration (bit-depth, channels, pixel-parallelism) the xf::cv::Mat was
created. Array2xfMat supports line stride. Line stride is the number of pixels
which needs to be added to the address in the first pixel of a row in order to access the 
first pixel of the next row.

.. code:: c

   //Without Line stride support
   template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
   void Array2xfMat(ap_uint< PTR_WIDTH > *srcPtr, xf::cv::Mat<MAT_T,ROWS,COLS,NPC>& dstMat)
   
   //With Line stride support
   template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
   void Array2xfMat(ap_uint< PTR_WIDTH > *srcPtr, xf::cv::Mat<MAT_T,ROWS,COLS,NPC>& dstMat, int stride)

.. table:: Table. Array2xfMat Parmater Description

   +-----------------------------------+-----------------------------------+
   | Parameter                         | Description                       |
   +===================================+===================================+
   | PTR_WIDTH                         | Data width of the input pointer.  |
   |                                   | The value must be power 2,        |
   |                                   | starting from 8 to 512.           |
   +-----------------------------------+-----------------------------------+
   | MAT_T                             | Input Mat type. Example XF_8UC1,  |
   |                                   | XF_16UC1, XF_8UC3 and XF_8UC4     |
   +-----------------------------------+-----------------------------------+
   | ROWS                              | Maximum height of image           |
   +-----------------------------------+-----------------------------------+
   | COLS                              | Maximum width of image            |
   +-----------------------------------+-----------------------------------+
   | NPC                               | Number of pixels computed in      |
   |                                   | parallel. Example XF_NPPC1,       |
   |                                   | XF_NPPC8                          |
   +-----------------------------------+-----------------------------------+
   | srcPtr                            | Input pointer. Type of the        |
   |                                   | pointer based on the PTR_WIDTH.   |
   +-----------------------------------+-----------------------------------+
   | dstMat                            | Output image of type xf::cv::Mat  |
   +-----------------------------------+-----------------------------------+
   | stride                            | Line stride.                      |
   |                                   | Default value is dstMat.cols      |
   +-----------------------------------+-----------------------------------+


xfMat2Array
~~~~~~~~~~~

This function converts the input xf::cv::Mat to output array. The output of
the xf::kernel function will be xf::cv::Mat, and it will require to convert
that to output pointer. xfMat2Array supports line stride. Line stride is the number of pixels
which needs to be added to the address in the first pixel of a row in order to access the 
first pixel of the next row.

.. code:: c

   //Without Line stride support
   template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC, int FILLZERO = 1>
   void xfMat2Array(xf::cv::Mat<MAT_T,ROWS,COLS,NPC>& srcMat, ap_uint< PTR_WIDTH > *dstPtr)
   
   //With Line stride support
   template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC, int FILLZERO = 1>
   void xfMat2Array(xf::cv::Mat<MAT_T,ROWS,COLS,NPC>& srcMat, ap_uint< PTR_WIDTH > *dstPtr, int stride)
   
.. table:: Table . xfMat2Array Parameter Description

   +-----------------------------------+-----------------------------------+
   | Parameter                         | Description                       |
   +===================================+===================================+
   | PTR_WIDTH                         | Data width of the output pointer. |
   |                                   | The value must be power 2, from 8 |
   |                                   | to 512.                           |
   +-----------------------------------+-----------------------------------+
   | MAT_T                             | Input Mat type. Example XF_8UC1,  |
   |                                   | XF_16UC1, XF_8UC3 and XF_8UC4     |
   +-----------------------------------+-----------------------------------+
   | ROWS                              | Maximum height of image           |
   +-----------------------------------+-----------------------------------+
   | COLS                              | Maximum width of image            |
   +-----------------------------------+-----------------------------------+
   | NPC                               | Number of pixels computed in      |
   |                                   | parallel. Example XF_NPPC1,       |
   |                                   | XF_NPPC8                          |
   +-----------------------------------+-----------------------------------+
   | FILLZERO                          | Line padding Flag. Use when line  |
   |                                   | stride support is needed.         |
   |                                   | Default value is 1                |   
   +-----------------------------------+-----------------------------------+
   | dstPtr                            | Output pointer. Type of the       |
   |                                   | pointer based on the PTR_WIDTH.   |
   +-----------------------------------+-----------------------------------+
   | srcMat                            | Input image of type xf::cv::Mat   |
   +-----------------------------------+-----------------------------------+
   | stride                            | Line stride.                      |
   |                                   | Default value is srcMat.cols      |
   +-----------------------------------+-----------------------------------+

Interface pointer widths
~~~~~~~~~~~~~~~~~~~~~~~~

Minimum pointer widths for different configurations is shown in the
following table:

.. table:: Table . Minimum and maximum pointer widths for different Mat types

   +-----------------+-----------------+-----------------+-----------------+
   | MAT type        | Parallelism     | Min PTR_WIDTH   | Max PTR_WIDTH   |
   +=================+=================+=================+=================+
   | XF_8UC1         | XF_NPPC1        | 8               | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF_16UC1        | XF_NPPC1        | 16              | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF\_ 8UC1       | XF_NPPC8        | 64              | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF\_ 16UC1      | XF_NPPC8        | 128             | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF\_ 8UC3       | XF_NPPC1        | 32              | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF\_ 8UC3       | XF_NPPC8        | 256             | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF_8UC4         | XF_NPPC8        | 256             | 512             |
   +-----------------+-----------------+-----------------+-----------------+
   | XF_8UC3         | XF_NPPC16       | 512             | 512             |
   +-----------------+-----------------+-----------------+-----------------+

Kernel-to-Kernel streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two utility functions available in Vitis Vision, axiStrm2xfMat and xfMat2axiStrm to support streaming 
of data between two kernels. For more details on kernel-to-kernel streaming, refer to the "Streaming Data Transfers Between the
Kernels" section of [UG1393](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2021_1/ug1393-vitis-application-acceleration.pdf) document.

axiStrm2xfMat
~~~~~~~~~~~~~

axiStrm2xfMat is used by consumer kernel to support streaming data transfer between two kernels. 
Consumer kernel receives data from producer kernel through kernel streaming interface which is defined by hls:stream 
with the ap_axiu< PTR_WIDTH, 0, 0, 0> data type. axiStrm2xfMat would read from AXI stream and write into xf::cv:Mat based 
on particular configuration (bit-depth, channels, pixel-parallelism) the xf::cv:Mat was created.


.. code:: c

   template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
   void axiStrm2xfMat(hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& srcPtr, xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& dstMat)

.. table:: Table . Parameter description of axiStrm2xfMat function


   +-----------------+-------------------------------------------------------------------------------------+
   | Parameter	     |  Description                                                                        | 
   +=================+=====================================================================================+
   | PTR_WIDTH	     | 	Data width of the input pointer. The value must be power 2, starting from 8 to 512.|
   +-----------------+-------------------------------------------------------------------------------------+
   | MAT_T           |  Input Mat type. Example XF_8UC1, XF_16UC1, XF_8UC3 and XF_8UC4                     |
   +-----------------+-------------------------------------------------------------------------------------+
   | ROWS            |  Maximum height of image                                                            |
   +-----------------+-------------------------------------------------------------------------------------+
   | COLS            |  Maximum width of image                                                             |
   +-----------------+-------------------------------------------------------------------------------------+
   | NPC             |  Number of pixels computed in parallel. Example XF_NPPC1, XF_NPPC8                  |
   +-----------------+-------------------------------------------------------------------------------------+
   | srcPtr          |  Input image of type hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >                      |
   +-----------------+-------------------------------------------------------------------------------------+
   | dstMat          |  Output image of type xf::cv::Mat                                                   |
   +-----------------+-------------------------------------------------------------------------------------+

xfMat2axiStrm
~~~~~~~~~~~~~

xfMat2axiStrm is used by producer kernel to support streaming data transfer between two kernels. 
This function converts the input xf:cv::Mat to AXI stream based on particular configuration (bit-depth, channels, pixel-parallelism). 

.. code:: c

   template <int PTR_WIDTH, int MAT_T, int ROWS, int COLS, int NPC>
   void xfMat2axiStrm(xf::cv::Mat<MAT_T, ROWS, COLS, NPC>& srcMat, hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >& dstPtr)

.. table:: Table . Parameter description of xfMat2axiStrm function


   +-----------------+-------------------------------------------------------------------------------------+
   | Parameter	     |  Description                                                                        | 
   +=================+=====================================================================================+
   | PTR_WIDTH       | 	Data width of the input pointer. The value must be power 2, starting from 8 to 512.|
   +-----------------+-------------------------------------------------------------------------------------+
   | MAT_T           |  Input Mat type. Example XF_8UC1, XF_16UC1, XF_8UC3 and XF_8UC4                     |
   +-----------------+-------------------------------------------------------------------------------------+
   | ROWS            |  Maximum height of image                                                            |
   +-----------------+-------------------------------------------------------------------------------------+
   | COLS            |  Maximum width of image                                                             |
   +-----------------+-------------------------------------------------------------------------------------+
   | NPC             |  Number of pixels computed in parallel. Example XF_NPPC1, XF_NPPC8                  |
   +-----------------+-------------------------------------------------------------------------------------+
   | srcPtr          |  Input image of type hls::stream<ap_axiu<PTR_WIDTH, 0, 0, 0> >                      |
   +-----------------+-------------------------------------------------------------------------------------+
   | dstMat          |  Output image of type xf::cv::Mat                                                   |
   +-----------------+-------------------------------------------------------------------------------------+


Memory Mapped Kernels
~~~~~~~~~~~~~~~~~~~~~

In the memory map based kernels such as crop, Mean-shift tracking and
bounding box, the input read will be for particular block of memory
based on the requirement for the algorithm. The streaming interfaces
will require the image to be read in raster scan manner, which is not
the case for the memory mapped kernels. The methodology to handle this
case is as follows:

.. code:: c

   extern “C” 
   { 
   void func_top (ap_uint *gmem_in, ap_uint *gmem_out, ...) { 
   xf::cv::Mat<…> in_mat(…,gmem_in), out_mat(…,gmem_out);
   xf::cv::kernel<…> (in_mat, out_mat…); 
   }
   }

The gmem pointers must be mapped to the xf::cv::Mat objects during the
object creation, and then the memory mapped kernels are called with
these mats at the interface. It is necessary that the pointer size must
be same as the size required for the xf::Vitis-Vision-func, unlike the
streaming method where any higher size of the pointers (till 512-bits)
are allowed.


Makefile
---------

Examples for makefile are provided in the examples and tests section of GitHub.


Design example Using Library on Vitis
-------------------------------------

Following is a multi-kernel example, where different kernel runs
sequentially in a pipeline to form an application. This example performs
Canny edge detection, where two kernels are involved, Canny and edge
tracing. Canny function will take gray-scale image as input and provided
the edge information in 3 states (weak edge (1), strong edge (3), and
background (0)), which is being fed into edge tracing, which filters out
the weak edges. The prior works in a streaming based implementation and
the later in a memory mapped manner.

Host code
~~~~~~~~~

The following is the Host code for the canny edge detection example. The
host code sets up the OpenCL platform with the FPGA of processing
required data. In the case of Vitis Vision example, the data is an image.
Reading and writing of images are enabled using called to functions from
Vitis Vision.

.. code:: c

   // setting up device and platform
       std::vector<cl::Device> devices = xcl::get_xil_devices();
       cl::Device device = devices[0];
       cl::Context context(device);
       cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);
       std::string device_name = device.getInfo<CL_DEVICE_NAME>();

       // Kernel 1: Canny
       std::string binaryFile=xcl::find_binary_file(device_name,"krnl_canny");
       cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
       devices.resize(1);
       cl::Program program(context, devices, bins);
       cl::Kernel krnl(program,"canny_accel");

       // creating necessary cl buffers for input and output
       cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY,(height*width));
       cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY,(height*width/4));


       // Set the kernel arguments
       krnl.setArg(0, imageToDevice);
       krnl.setArg(1, imageFromDevice);
       krnl.setArg(2, height);
       krnl.setArg(3, width);
       krnl.setArg(4, low_threshold);
       krnl.setArg(5, high_threshold);

       // write the input image data from host to device memory
       q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0,(height*(width)),img_gray.data);
       // Profiling Objects
       cl_ulong start= 0;
       cl_ulong end = 0;
       double diff_prof = 0.0f;
       cl::Event event_sp;

       // Launch the kernel
       q.enqueueTask(krnl,NULL,&event_sp);
       clWaitForEvents(1, (const cl_event*) &event_sp);

       // profiling
       event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
       event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
       diff_prof = end-start;
       std::cout<<(diff_prof/1000000)<<"ms"<<std::endl;

       // Kernel 2: edge tracing
       cl::Kernel krnl2(program,"edgetracing_accel");

       cl::Buffer imageFromDeviceedge(context, CL_MEM_WRITE_ONLY,(height*width));

       // Set the kernel arguments
       krnl2.setArg(0, imageFromDevice);
       krnl2.setArg(1, imageFromDeviceedge);
       krnl2.setArg(2, height);
       krnl2.setArg(3, width);
       
       // Profiling Objects
       cl_ulong startedge= 0;
       cl_ulong endedge = 0;
       double diff_prof_edge = 0.0f;
       cl::Event event_sp_edge;

       // Launch the kernel
       q.enqueueTask(krnl2,NULL,&event_sp_edge);
       clWaitForEvents(1, (const cl_event*) &event_sp_edge);

       // profiling
       event_sp_edge.getProfilingInfo(CL_PROFILING_COMMAND_START,&startedge);
       event_sp_edge.getProfilingInfo(CL_PROFILING_COMMAND_END,&endedge);
       diff_prof_edge = endedge-startedge;
       std::cout<<(diff_prof_edge/1000000)<<"ms"<<std::endl;

       
       //Copying Device result data to Host memory
       q.enqueueReadBuffer(imageFromDeviceedge, CL_TRUE, 0,(height*width),out_img_edge.data);
       q.finish();

Top level kernel
~~~~~~~~~~~~~~~~~

Below is the top-level/wrapper function with all necessary glue logic.

.. code:: c

   // streaming based kernel
   #include "xf_canny_config.h"

   extern "C" {
   void canny_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out, int rows, int cols,int low_threshold,int high_threshold)
   {
   #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
   #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
   #pragma HLS INTERFACE s_axilite port=img_inp  bundle=control
   #pragma HLS INTERFACE s_axilite port=img_out  bundle=control

   #pragma HLS INTERFACE s_axilite port=rows     bundle=control
   #pragma HLS INTERFACE s_axilite port=cols     bundle=control
   #pragma HLS INTERFACE s_axilite port=low_threshold     bundle=control
   #pragma HLS INTERFACE s_axilite port=high_threshold     bundle=control
   #pragma HLS INTERFACE s_axilite port=return   bundle=control

       xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, INTYPE> in_mat(rows,cols);
       
       xf::cv::Mat<XF_2UC1, HEIGHT, WIDTH, XF_NPPC32> dst_mat(rows,cols);
       
       #pragma HLS DATAFLOW 

       xf::cv::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT,WIDTH,INTYPE>(img_inp,in_mat);
       xf::cv::Canny<FILTER_WIDTH,NORM_TYPE,XF_8UC1,XF_2UC1,HEIGHT, WIDTH,INTYPE,XF_NPPC32,XF_USE_URAM>(in_mat,dst_mat,low_threshold,high_threshold);
       xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH,XF_2UC1,HEIGHT,WIDTH,XF_NPPC32>(dst_mat,img_out);
       
       
   }
   }
   // memory mapped kernel
   #include "xf_canny_config.h"
   extern "C" {
   void edgetracing_accel(ap_uint<INPUT_PTR_WIDTH> *img_inp, ap_uint<OUTPUT_PTR_WIDTH> *img_out, int rows, int cols)
   {
   #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem3
   #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem4
   #pragma HLS INTERFACE s_axilite port=img_inp  bundle=control
   #pragma HLS INTERFACE s_axilite port=img_out  bundle=control

   #pragma HLS INTERFACE s_axilite port=rows     bundle=control
   #pragma HLS INTERFACE s_axilite port=cols     bundle=control
   #pragma HLS INTERFACE s_axilite port=return   bundle=control


       xf::cv::Mat<XF_2UC1, HEIGHT, WIDTH, XF_NPPC32> _dst1(rows,cols,img_inp);
       xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC8> _dst2(rows,cols,img_out);
       xf::cv::EdgeTracing<XF_2UC1,XF_8UC1,HEIGHT, WIDTH, XF_NPPC32,XF_NPPC8,XF_USE_URAM>(_dst1,_dst2);
       
   }
   }


Evaluating the Functionality
=============================

You can build the kernels and test the functionality through software
emulation, hardware emulation, and running directly on a supported
hardware with the FPGA. Use the following
commands to setup the basic environment:

.. code:: c

   $ cd <path to the folder where makefile is present>
   $ source <path to the Vitis installation folder>/Vitis/<version number>/settings64.sh
   $ export DEVICE=<path-to-platform-directory>/<platform>.xpfm

For PCIe devices, set the following:

.. code:: c

   $ source <path to Xilinx_xrt>/setup.sh
   
   $ export OPENCV_INCLUDE=< path-to-opencv-include-folder >
   
   $ export OPENCV_LIB=< path-to-opencv-lib-folder >
   
   $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >

For embedded devices, set the following:

.. code:: c

   Download the platform, and common-image from Xilinx Download Center. Run the sdk.sh script from the common-image directory to install sysroot using the command : 
   $ ./sdk.sh -y -d ./ -p

   Unzip the rootfs file : 
   $ gunzip ./rootfs.ext4.gz

   $ export SYSROOT=< path-to-platform-sysroot >
   
   $ export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >
   
   $ export PERL=<path-to-perl-installation-location> #For example, "export PERL=/usr/bin/perl". Please make sure that Expect.pm package is available in your Perl installation.
   
Software Emulation
------------------

Software emulation is equivalent to running a C-simulation of the
kernel. The time for compilation is minimal, and is therefore
recommended to be the first step in testing the kernel. Following are
the steps to build and run for the software emulation:

*For PCIe devices:*

.. code:: c

   $ make host xclbin TARGET=sw_emu
   
   $ make run TARGET=sw_emu
   
*For embedded devices:*

.. code:: c

   $ make host xclbin TARGET=sw_emu HOST_ARCH=< aarch32 | aarch64 >

   $ make run TARGET=sw_emu HOST_ARCH=< aarch32 | aarch64 >


Hardware Emulation
-------------------

Hardware emulation runs the test on the generated RTL after synthesis of
the C/C++ code. The simulation, since being done on RTL requires longer
to complete when compared to software emulation. Following are the steps
to build and run for the hardware emulation:

*For PCIe devices:*

.. code:: c

   $ make host xclbin TARGET=hw_emu
   
   $ make run TARGET=hw_emu
   
*For embedded devices:*


.. code:: c

   
   $ make host xclbin TARGET=hw_emu HOST_ARCH=< aarch32 | aarch64 >
   
   $ make run TARGET=hw_emu HOST_ARCH=< aarch32 | aarch64 >


Testing on the Hardware
------------------------

To test on the hardware, the kernel must be compiled into a bitstream
(building for hardware). This would consume some time since the C/C++ code must be converted to
RTL, run through synthesis and implementation process before a bitstream
is created. As a prerequisite the drivers has to be installed for
corresponding XSA, for which the example was built for. Following are
the steps to build the kernel and run on a hardware:

*For PCIe devices:*

.. code:: c

   $ make host xclbin TARGET=hw
   
   $ make run TARGET=hw
   
*For embedded devices:*

.. code:: c

   $ make host xclbin TARGET=hw HOST_ARCH=< aarch32 | aarch64 >
   
   $ make run TARGET=< sw_emu|hw_emu|hw > HOST_ARCH=< aarch32 | aarch64 > #This command will generate only the sd_card folder in case of hardware build.

**Note**. For hw run on embedded devices, copy the generated sd_card folder content under package_hw to an SD Card. More information on preparing the SD Card is available [here](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842385/How+to+format+SD+card+for+SD+boot#HowtoformatSDcardforSDboot-CopingtheImagestotheNewPartitions). After successful booting of the board, run the following commands:

    cd /mnt

    export XCL_BINDIR=< xclbin-folder-present-in-the-sd_card > #For example, "export XCL_BINDIR=xclbin_zcu102_base_hw"

    ./run_script.sh



