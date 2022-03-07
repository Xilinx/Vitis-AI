Vitis AIE Design Methodology
============================

Following are critical components in making a kernel work on a platform using Vitis™:

#. Prepare the Kernels
#. Data Flow Graph construction
#. Setting up platform ports
#. Host code integration
#. Makefile to compile the kernel for x86 simulation / aie simulation / hw-emulation / hw runs

Prepare the Kernels
-------------------

Kernels are computation functions that form the fundamental building blocks of the data flow graph specifications. Kernels are declared as ordinary C/C++ functions that return void and can use special data types as arguments (discussed in `Window and Streaming Data API`_). Each kernel should be defined in its own source file. This organization is recommended for reusable and faster compilation. Furthermore, the kernel source files should include all relevant header files to allow for independent compilation. It is recommended that a header file (kernels.h in this documentation) should declare the function prototypes for all kernels used in a graph. An example is shown below.

.. code:: c

    #ifndef _KERNELS_16B_H_
    #define _KERNELS_16B_H_

    #include <adf/stream/types.h>
    #include <adf/window/types.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define PARALLEL_FACTOR_16b 16 // Parallelization factor for 16b operations (16x mults)
    #define SRS_SHIFT 10           // SRS shift used can be increased if input data likewise adjusted)

    void filter2D(input_window_int16* input, const int16_t (&coeff)[16], output_window_int16* output);

    #endif

:ref:`Vitis Vision AIE library functions <aie_library_functions>` packaged with Vitis Vision AIE library are pre optimized vector implementations for various computer vision tasks. These functions can be directly included in user kernel (as shown in example below)

.. code:: c

    #include "imgproc/xf_filter2d_16b_aie.hpp"
    #include "kernels.h"

    void filter2D(input_window_int16* input, const int16_t (&coeff)[16], output_window_int16* output) {
            xf::cv::aie::filter2D_k3_border(input, coeff, output);
    };

.. _Window and Streaming Data API: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/window_streaming_api.html#efv1509388613160

Data Flow Graph construction
----------------------------

Once AIE kernels have been prepared, next step is to create a Data Flow Graph class which defines the top level ports, `Run time parameters`_, connectivity, constraints etc. This consists of below steps

#. Create graph.h and include Adaptive Data Flow (ADF) header file (adf.h). Also include header file with kernel function prototypes (kernel.h)

   .. code:: c

      #include <adf.h>
      #include "kernels.h"

#. Define your graph class by using the objects which are defined in the adf name space. All user graphs are derived from the class graph.

   .. code:: c

      include <adf.h>
      #include "kernels.h"
     
      using namespace adf;
     
      class myGraph : public graph {
      private:
          kernel k1;
      };

#. Add top level ports to graph. These ports will be responsible to data transfers to / from the kernels.

   .. code:: c

      #include <adf.h>
      #include "kernels.h"
   
      using namespace adf;
   
      class simpleGraph : public graph {
      private:
          kernel k1;
   
      public:
          port<input> inptr;
          port<output> outptr;
          port<input> kernelCoefficients;
      };


#. Specify connections of top level ports to kernels. Primary connections type are `Window`_, `Stream`_, `Run time parameters`_. Below is example code specifying connectivity.

   .. code:: c

      class myGraph : public adf::graph {
      private:
          kernel k1;
      public:
          port<input> inptr;
          port<output> outptr;
          port<input> kernelCoefficients;
   
          myGraph() {
              k1 = kernel::create(filter2D);
              adf::connect<window<TILE_WINDOW_SIZE> >(inptr, k1.in[0]);
              adf::connect<parameter>(kernelCoefficients, async(k1.in[1]));
              adf::connect<window<TILE_WINDOW_SIZE> >(k1.out[0], outptr);
          }
      };

#. Specify source file location and other constraints for each kernel

   .. code:: c

      class myGraph : public adf::graph {
      private:
          kernel k1;
      public:
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
          }
      };

.. _Run time parameters: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/runtime_graph_api.html#xpw1512590673863
.. _Window: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/window_streaming_api.html#iii1512524581327
.. _Stream: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/window_streaming_api.html#xta1512524637575


Setting up platform ports
-------------------------

Next step is to create a graph.cpp file with platform ports and virtual platform specification. A virtual platform specification helps to connect the data flow graph written with external I/O mechanisms specific to the chosen target for testing or eventual deployment. The platform could be specified for a simulation, emulation, or an actual hardware execution target.

.. code:: c

   simulation::platform<inputs, outputs> platform_name(port_attribute_list);

There are 3 types of platform ports attributes which describe how data is transferred to / from AIE cores.

.. _fileio_aie:

FileIO
~~~~~~

By default, a platform port attribute is a string name used to construct an attribute of type FileIO. The string specifies the name of an input or output file relative to the current directory that will source or sink the platform data. The explicit form is specified in the following example using a FileIO constructor.

.. code:: c

   FileIO* in = new FileIO(input_file_name);
   FileIO* out = new FileIO(output_file_name);
   simulation::platform<1,1> plat(in,out);

FileIO ports are solely for the purpose of application simulation in the absence of an actual hardware platform. They are provided as a matter of convenience to test out a data flow graph in isolation before it is connected to a real platform. An actual hardware platform exports either stream or memory ports.

.. _plio_aie:

PLIO
~~~~

A PLIO port attribute is used to make external stream connections that cross the AI Engine to programmable logic (PL) boundary. The following example shows how the PLIO attributes shown in the previous table can be used in a program to read input data from a file or write output data to a file. The PLIO width and frequency of the PLIO port are also provided in the PLIO constructor. For more details please refer `PLIO Attributes`_.

.. code:: c

   //Virtual platform ports
   PLIO* in1 = new PLIO("DataIn1", adf::plio_64_bits, "data/input.txt");
   PLIO* out1 = new PLIO("DataOut1", adf::plio_64_bits, "data/output.txt");
   simulation::platform<1, 1> platform(in1,out1);

   //Graph object
   myGraph filter_graph;

   //Virtual platform connectivity
   connect<> net0(platform.src[0], filter_graph.inptr);
   connect<> net1(filter_graph.outptr, platform.sink[0]);

.. _PLIO Attributes: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/use_virtual_program.html#bna1512607665307

.. _gmio_aie:

GMIO
~~~~

A GMIO port attribute is used to make external memory-mapped connections to or from the global memory. These connections are made between an AI Engine graph and the logical global memory ports of a hardware platform design. For more details please refer `GMIO Attributes`_.

.. code:: c

   GMIO gmioIn1("gmioIn1", 64, 1000);
   GMIO gmioOut("gmioOut", 64, 1000);
   simulation::platform<1, 1> platform(&gmioIn1, &gmioOut);

   myGraph filter_graph;

   connect<> net0(platform.src[0], filter_graph.in1);
   connect<> net1(filter_graph.out1, platform.sink[0]);

.. _GMIO Attributes: https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/use_virtual_program.html#pxw1512607606711

Host code integration
---------------------

Depending upon the functional verification model used, the top level application can be written using on of 2 ways.

x86Simulation / AIE simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this mode the top level application can be written inside graph.cpp file. The application contains an instance of ADF graph and a main function within which API's are called to initialize, run and end the graph. It may also have additional API's to update `Run time parameters`_. Additionally for hw emulation / hw run modes, the 'main()' function can be guarded by a #ifdef to ensure graph is only initialized once, or run only once. The following example code is the simple application defined in `Creating a Data Flow Graph (Including Kernels)`_ with the additional guard macro __AIESIM__ and __X86SIM__.

.. code:: c

   // Virtual platform ports
   PLIO* in1 = new PLIO("DataIn1", adf::plio_64_bits, "data/input.txt");
   PLIO* out1 = new PLIO("DataOut1", adf::plio_64_bits, "data/output.txt");
   simulation::platform<1, 1> platform(in1, out1);

   // Graph object
   myGraph filter_graph;

   // Virtual platform connectivity
   connect<> net0(platform.src[0], filter_graph.inptr);
   connect<> net1(filter_graph.outptr, platform.sink[0]);

   #define SRS_SHIFT 10
   float kData[9] = {0.0625, 0.1250, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};


   #if defined(__AIESIM__) || defined(__X86SIM__)
   int main(int argc, char** argv) {
       filter_graph.init();
       filter_graph.update(filter_graph.kernelCoefficients, float2fixed_coeff<10, 16>(kData).data(), 16);
       filter_graph.run(1);
       filter_graph.end();
       return 0;
   }
   #endif

In case GMIO based ports are used

.. code:: c

   #if defined(__AIESIM__) || defined(__X86SIM__)
   int main(int argc, char** argv) {
       ...
       ...
       int16_t* inputData = (int16_t*)GMIO::malloc(BLOCK_SIZE_in_Bytes);
       int16_t* outputData = (int16_t*)GMIO::malloc(BLOCK_SIZE_in_Bytes);

       //Prepare input data
       ...
       ...

       filter_graph.init();
       filter_graph.update(filter_graph.kernelCoefficients, float2fixed_coeff<10, 16>(kData).data(), 16);

       filter_graph.run(1);

       //GMIO Data transfer calls
       gmioIn[0].gm2aie_nb(inputData, BLOCK_SIZE_in_Bytes);
       gmioOut[0].aie2gm_nb(outputData, BLOCK_SIZE_in_Bytes);
       gmioOut[0].wait();

       printf("after grph wait\n");
       filter_graph.end();

       ...
   }
   #endif

.. _Creating a Data  Flow Graph (Including Kernels): https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/ai_engine_programming.html#tzk1513812699928

HW emulation / HW run
~~~~~~~~~~~~~~~~~~~~~

For x86Simulation / AIE simulation, top level application had simple ADF API calls to initialize / run / end the graph. However, for actual AI Engine graph applications the host code must do much more than those simple tasks. The top-level PS application running on the Cortex®-A72, controls the graph and PL kernels: manage data inputs to the graph, handle data outputs from the graph, and control any PL kernels working with the graph. Sample code is illustrated below


.. code:: c

   1.// Open device, load xclbin, and get uuid
       
   auto dhdl = xrtDeviceOpen(0);//device index=0

   xrtDeviceLoadXclbinFile(dhdl,xclbinFilename);
   xuid_t uuid;
   xrtDeviceGetXclbinUUID(dhdl, uuid);
   adf::registerXRT(dhdl, uuid);

   2. Allocate output buffer objects and map to host memory

   xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, output_size_in_bytes, 0, /*BANK=*/0);
   std::complex<short> *host_out = (std::complex<short>*)xrtBOMap(out_bohdl);

   3. Get kernel and run handles, set arguments for kernel, and launch kernel.
   xrtKernelHandle s2mm_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm"); // Open kernel handle
   xrtRunHandle s2mm_rhdl = xrtRunOpen(s2mm_khdl); 
   xrtRunSetArg(s2mm_rhdl, 0, out_bohdl); // set kernel arg
   xrtRunSetArg(s2mm_rhdl, 2, OUTPUT_SIZE); // set kernel arg
   xrtRunStart(s2mm_rhdl); //launch s2mm kernel

   // ADF API:run, update graph parameters (RTP) and so on
   gr.init();
   gr.update(gr.size, 1024);//update RTP
   gr.run(16);//start AIE kernel
   gr.wait();

   4. Wait for kernel completion.
   auto state = xrtRunWait(s2mm_rhdl);

   5. Sync output device buffer objects to host memory.

   xrtBOSync(out_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE , output_size_in_bytes,/*OFFSET=*/ 0);

   //6. post-processing on host memory - "host_out

:ref:`Vitis Vision AIE library functions <aie_library_functions>` provide optimal vector implementations of various computer vision algorithms. These functions are expected to process high resolution images. However because local memory of AIE core module is limited, entire image can't be fit into it. Also accessing DDR for reading / writing image data will be highly inefficient both for performance and power. To overcome this limitation host code is expected to split the high resolution image into smaller tiles which fit in AIE Engine local memory in ping-pong fashion. Splitting of high resolution image in smaller tiles is a complex operation as it need to be aware of overlap regions and borders. Also the tile size is expected to be aligned with vectorization factor of the kernel.

To facilitate this Vitis Vision Library provides data movers which perform smart tiling / stitching of high resolution images which can meet all above requirements. There are two versions made available which can provide data movement capabilities both using PLIO and GMIO interfaces. A high level class abstraction is provided with simple API interface to facilitate data transfers. The class abstraction allows seamless transition between PLIO - GMIO methods of data transfers.

.. Important::
   **For HW emulation / HW run it is imperative to include graph.cpp inside host.cpp. This is because platform port specification and ADF graph object instance is declared in graph.cpp.**

.. _xfcvdatamovers_aie:

xfcvDataMovers
~~~~~~~~~~~~~~

xfcvDataMovers class provides a high level API abstraction to initiate data transfer from DDR to AIE core and vice versa for hw-emulation / hw runs. Because each AIE core has limited amount of local memory which is not sufficient to fit in entire high resolution images (input / output), each image needs to be partitioned into smaller tiles and then send to AIE core for computation. After computation the tiled image at output is stitched back to generate the high resolution image at the output. This process involves complex computation as tiling needs to ensure proper border handling and overlap processing in case of
convolution based kernels.

xfcvDataMovers class object takes input some simple parameters from users and provides a simple data transaction API where user does not have to bother about the complexity. Moreover it provides a template parameter using which application can switch from PL based data movement to GMIO based (and vice versa) seamlessly.

.. csv-table:: Table. xfcvDataMovers Template Parameters
   :file: tables/xfcvDataMoversTemplate.csv
   :widths: 20, 50

.. csv-table:: Table. xfcvDataMovers constructor parameters
   :file: tables/xfcvDataMoversCtor.csv
   :widths: 20, 50

.. note::
   Horizontal overlap and Vertical overlaps should be computed for the complete pipeline. For example if the pipeline has a single 3x3 2D filter then overlap sizes (both horizontal and vertical) will be 1. However in case of two such filter operations which are back to back the overlap size will be 2. Currently if it is expected from users to provide this input correctly.

The data transfer using xfcvDataMovers class can be done in one out of 2 ways.

#. PLIO data movers

   This is the default mode for xfcvDataMovers class operation. When this method is used, data is transferred using hardware Tiler / Stitcher IPs provided by Xilinx. The :ref:`Makefile <aie_makefile>` provided with designs examples shipped with the library provide location to .xo files for these IP's. It also shows how to incorporate them in Vitis Build System. Having said that, user needs to create an object of xfcvDataMovers class per input / output image as shown in code below

   .. Important::
      **The implementations of Tiler and Stitcher for PLIO, are provided as .xo files in 'L1/lib/hw' folder. By using these files, you are agreeing to the terms and conditions specified in the LICENSE.txt file available in the same directory.**

   .. code:: c

      int overlapH = 1;
      int overlapV = 1;
      xF::xfcvDataMovers<xF::TILER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR> tiler(overlapH, overlapV);
      xF::xfcvDataMovers<xF::STITCHER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR> stitcher;

   Choice of MAX_TILE_HEIGHT / MAX_TILE_WIDTH provide constraints on image tile size which in turn governs local memory usage. The image tile size in bytes can be computed as below

   Image tile size = (TILE_HEADER_SIZE_IN_BYTES + MAX_TILE_HEIGHT*MAX_TILE_WIDTH*sizeof(DATA_TYPE))

   Here TILE_HEADER_SIZE_IN_BYTES is 128 bytes for current version of Tiler / Stitcher. DATA_TYPE in above example is int16_t (2 bytes).o

   .. note::
      Current version of HW data movers have 8_16 configuration (i.e. 8 bit image element data type on host side and 16 bit image element data type on AIE kernel side). In future more such configurations will be provided (example: 8_8 / 16_16 etc.)

   Tiler / Stitcher IPs use PL resources available on VCK boards. For 8_16 configuration below table illustrates resource utilization numbers for theese IPs. The numbers correspond to single instance of each IP.

   .. table:: Tiler / Stitcher resource utilization (8_16 config)
      :widths: 10,15,15,15,15,15

      +----------------+--------+-------+-------+--------+---------+
      |                |  LUTs  |  FFs  | BRAMs |  DSPs  |   Fmax  |
      +================+========+=======+=======+========+=========+
      | **Tiler**      |  2761  |  3832 |   5   |   13   | 400 MHz |
      +----------------+--------+-------+-------+--------+---------+
      | **Stitcher**   |  2934  |  3988 |   5   |   7    | 400 MHz |
      +----------------+--------+-------+-------+--------+---------+
      | **Total**      |  5695  |  7820 |   10  |   20   |         |
      +----------------+--------+-------+-------+--------+---------+

#. GMIO data movers

   Transition to GMIO based data movers can be achieved by using a specialized template implementation of above class. All above constraints w.r.t Image tile size calculation are valid here as well. Sample code is shown below

   .. code:: c

      xF::xfcvDataMovers<xF::TILER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> tiler(1, 1);
      xF::xfcvDataMovers<xF::STITCHER, int16_t, MAX_TILE_HEIGHT, MAX_TILE_WIDTH, VECTORIZATION_FACTOR, 1, 0, true> stitcher;

   .. note::
      Last template parameter is set  to true, implying GMIO specialization.

Once the objects are constructed, simple API calls can be made to initiate the data transfers. Sample code is shown below

.. code:: c

   //For PLIO
   auto tiles_sz = tiler.host2aie_nb(src_hndl, srcImageR.size());
   stitcher.aie2host_nb(dst_hndl, dst.size(), tiles_sz);

   //For GMIO
   auto tiles_sz = tiler.host2aie_nb(srcData.data(), srcImageR.size(), {"gmioIn[0]"});
   stitcher.aie2host_nb(dstData.data(), dst.size(), tiles_sz, {"gmioOut[0]"});

.. note::
   GMIO data transfers take additional argument which is corresponding GMIO port to be used.

.. note::
   For GMIO based transfers there is a blocking method as well (host2aie(...) / aie2host(...)). For PLIO based data transfers the method only non-blocking API calls are provided.

Using 'tile_sz' user can run the graph appropriate number of times.

.. code:: c

   filter_graph.run(tiles_sz[0] * tiles_sz[1]);

After the runs are started, user needs to wait for all transactions to get complete.

.. code:: c

   filter_graph.wait();
   tiler.wait();
   stitcher.wait();

.. note::
   Current implementation of xfcvDataMovers support only 1 core. Multi core support is planned for future releases.
