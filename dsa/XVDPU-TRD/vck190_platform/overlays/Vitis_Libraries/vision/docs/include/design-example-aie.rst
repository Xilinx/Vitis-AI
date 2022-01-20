Design example Using Vitis Vision AIE Library
#############################################

Following example application performs a 2D filtering operation over a gray scale image. The
convolution kernel is a 3x3 window with floating point representation. The coefficients are
converted to fixed point representation before being passed to AIE core for computation. The
results are cross validated against OpenCV reference implementation. The example illustrates
both PLIO and GMIO based data transfers.

ADF Graph
=========

An AI Engine program consists of a data flow graph specification written in C++. The dataflow graph consists of top level ports, 
kernel instances and connectivity. a graph.h file is created which includes the header adf.h.

For more details on data flow graph creation, please refer `AI Engine Programming`_ .

.. _AI Engine Programming: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/ai_engine_programming.html 

.. code:: c

   #include "kernels.h"
   #include <adf.h>

   using namespace adf;

   class myGraph : public adf::graph {
      public:
       kernel k1;
       port<input> inptr;
       port<output> outptr;
       port<input> kernelCoefficients;

     myGraph() {
        k1 = kernel::create(filter2D);
        adf::connect<window<TILE_WINDOW_SIZE> >(inptr, k1.in[0]);
        adf::connect<parameter>(kernelCoefficients, async(k1.in[1]));
        adf::connect<window<TILE_WINDOW_SIZE> >(k1.out[0], outptr);

        source(k1) = "xf_filter2d.cc";
        // Initial mapping
        runtime<ratio>(k1) = 0.5;
     };
   }; 

Platform Ports
==============

A top-level application file graph.cpp is created which contains an instance of the graph class and is connected to a simulation platform. A virtual platform specification helps to connect the data flow graph written with external I/O
mechanisms specific to the chosen target for testing or eventual deployment.

.. code:: c

  #include "graph.h"

  // Virtual platform ports
  PLIO* in1 = new PLIO("DataIn1", adf::plio_64_bits, "data/input.txt");
  PLIO* out1 = new PLIO("DataOut1", adf::plio_64_bits, "data/output.txt");
  simulation::platform<1, 1> platform(in1, out1);

  // Graph object
  myGraph filter_graph;

  // Virtual platform connectivity
  connect<> net0(platform.src[0], filter_graph.inptr);
  connect<> net1(filter_graph.outptr, platform.sink[0]);


#. PLIO

   A PLIO port attribute is used to make external stream connections that cross the AI Engine to programmable logic (PL) boundary. PLIO attributes are used to specify the port name, port bit width and the input/output file names.
   Note that when simulating PLIO with data files, the data should be organized to accomodate both the width of the PL block as well as the data type of connecting port on the AI Engine block.

   .. code:: c

     //Platform ports
     PLIO* in1 = new PLIO("DataIn1", adf::plio_64_bits, "data/input.txt");
     PLIO* out1 = new PLIO("DataOut1", adf::plio_64_bits, "data/output.txt");


#. GMIO

   A GMIO port attribute is used to make external memory-mapped connections to or from the global memory. These connections are made between an AI Engine graph and the logical global memory ports of a hardware platform design.

   .. code:: c

      //Platform ports
      GMIO gmioIn1("gmioIn1", 64, 1000);
      GMIO gmioOut("gmioOut", 64, 1000);
   
      //Virtual platform
      simulation::platform<1, 1> platform(&gmioIn1, &gmioOut);
   
      //Graph object
      myGraph filter_graph;
   
      //Platform ports
      connect<> net0(platform.src[0], filter_graph.in1);
      connect<> net1(filter_graph.out1, platform.sink[0]);

Host code
=========

Host code 'host.cpp' will be running on the host processor which conatins the code to initialize and run the datamovers and the ADF graph. XRT APIs are
used to create the required buffers in the device memory. 

First a golden reference image is generated using OpenCV

.. code:: c

    int run_opencv_ref(cv::Mat& srcImageR, cv::Mat& dstRefImage, float coeff[9]) {
    cv::Mat tmpImage;
    cv::Mat kernel = cv::Mat(3, 3, CV_32F, coeff);
    cv::filter2D(srcImageR, dstRefImage, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    return 0;
    }

Then, xclbin is loaded on the device and the device handles are created

.. code:: c

   xF::deviceInit(xclBinName);

Buffers for input and output data are created using the XRT APIs and data from input CV::Mat is copied to the XRT buffer.

.. code:: c

        void* srcData = nullptr;
        xrtBufferHandle src_hndl = xrtBOAlloc(xF::gpDhdl, (srcImageR.total() * srcImageR.elemSize()), 0, 0);
        srcData = xrtBOMap(src_hndl);
        memcpy(srcData, srcImageR.data, (srcImageR.total() * srcImageR.elemSize()));

        // Allocate output buffer
        void* dstData = nullptr;
        xrtBufferHandle dst_hndl = xrtBOAlloc(xF::gpDhdl, (op_height * op_width * srcImageR.elemSize()), 0, 0);
        dstData = xrtBOMap(dst_hndl);
        cv::Mat dst(op_height, op_width, srcImageR.type(), dstData);

xfcvDataMovers objects tiler and stitcher are created. For more details on xfcvDataMovers refer :ref:`xfcvDataMovers <xfcvdatamovers_aie>`

.. code:: c

        xF::xfcvDataMovers<xF::TILER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR> tiler(1, 1);
        xF::xfcvDataMovers<xF::STITCHER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR> stitcher;

ADF graph is initialized and the filter coefficients are updated.        

.. code:: c

   filter_graph.init();
   filter_graph.update(filter_graph.kernelCoefficients, float2fixed_coeff<10, 16>(kData).data(), 16);

Metadata containing the tile information is generated.

.. code:: c

   tiler.compute_metadata(srcImageR.size());

The data transfer to AIE via datamovers is initiated along with graph run and further execution waits till the data transfer is complete.

.. code:: c

    auto tiles_sz = tiler.host2aie_nb(src_hndl, srcImageR.size());
    stitcher.aie2host_nb(dst_hndl, dst.size(), tiles_sz);

    std::cout << "Graph run(" << (tiles_sz[0] * tiles_sz[1]) << ")\n";

    filter_graph.run(tiles_sz[0] * tiles_sz[1]);

    filter_graph.wait();
    tiler.wait();
    stitcher.wait();

            


.. _aie_makefile:

Makefile
========

Run 'make help' to get list of commands and flows supported. Running below commands will initiate a hardware build.

.. code:: c

	source < path-to-Vitis-installation-directory >/settings64.sh
	export SYSROOT=< path-to-platform-sysroot >
	export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >
	make all TARGET=hw DEVICE=< path-to-platform-directory >/< platform >.xpfm
	
.. include:: include/f2d-l3-pipeline.rst
