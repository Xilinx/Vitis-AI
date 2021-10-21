..
   Copyright 2021 Xilinx, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _3_USING_EXAMPLES:

*******************
Using the Examples
*******************

=================================================
Compiling and Simulating Using the Example Design
=================================================

A Makefile is included with the example design. It is located inside the `L2/examples/fir_129t_sym/` directory. Use the following steps to compile, simulate and verify the example design using the Makefile.

#. Clean Work directory and all output files

   .. code-block::

         make clean

#. Compile the example design.

   .. code-block::

         make compile

#. Simulate the example design.

   .. code-block::

         make sim

   This generates the file `output.txt` in the `aiesimulator_output/data` directory.

#. To compare the output results with the golden reference, extract samples from output.txt, then perform a diff with respect to the reference using the following command.

   .. code-block::

         make check_op

#. Display the status summary with.

   .. code-block::

         make get_status

   This populates the status.txt file. Review this file to get the status.

   .. note:: All of the preceding steps are performed in sequence using the following command:

        .. code-block::

             make all

==================================================================
Using the Vitis Unified Software Platform to Run an Example Design
==================================================================

This section briefly describes how to create, build, and simulate a library element example using the Vitis |trade| integrated design environment (IDE).

Steps for Creating the Example Project in the Vitis IDE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fir129Example_system is a pre-packaged Vitis project (in compressed format) that can be downloaded here: `Fir129Example_system <https://www.xilinx.com/bin/public/openDownload?filename=examples.vitis-2021.1.Fir129Example_system_2021_06_24.ide.zip>`_. This can be imported into the Vitis IDE in a few steps, as described below.

1. Set the environment variable DSPLIB_ROOT. This must be set to the path to your directory where DSPLib is installed.

   .. code-block::

        setenv DSPLIB_ROOT <your-install-directory/dsplib>

2. Type `vitis` to launch the Vitis IDE, and create a new workspace.

3. Select **File → Import → Vitis project** exported zip file.

4. Click **Next** and browse to your download directory and select the downloaded ZIP file.

5. Set the correct path to the VCK190 platform file directory.

   a. In the Vitis IDE, expand the **Fir129Example [ aiengine ]** project and double-click the `Fir129Example.prj` file.

   b. On the **Platform Invalid** prompt, select **Add platform to repository**.

      |image7|

   c. Navigate to the installed vck190 platform file (.xpfm).

      .. note:: The VCK190 base platform must be downloaded from the Xilinx lounge.

6. Set the Active configuration to **Emulation-AIE** for the design.

   a. Right-click the **Fir129Example [ aiengine ]** and select **C/C++ Build Settings → Manage Configurations → Select Emulation-AIE → Set Active**.

      |image8|

   b. Click **OK**.

   c. Select **Apply and Close**.

7. Build the project by right-clicking the **Fir129Example [ aiengine]** project and select **Build Project**.

8. Perform simulation after the project is built by right-clicking the **Fir129Example [ aiengine ]** project and select **Run As → 1. Launch AIE Emulator**

   |image9|

9. After simulation is complete, navigate to **Fir129Example [ aiengine] → Emulation-AIE → aiesimulator_output → data** and compare output.txt with the fir_out_ref.txt file in the data folder.


====================
Example Graph Coding
====================

The following example can be used as a template for graph coding:

test.h
~~~~~~

.. code-block::

        #include <adf.h>
        #include "fir_sr_sym_graph.hpp"

        #define FIR129_LENGTH 129
        #define FIR129_SHIFT 15
        #define FIR129_ROUND_MODE 0
        #define FIR129_INPUT_SAMPLES 256

        using namespace adf ;
        namespace testcase {
            class test_kernel: public graph {
               private:
               // FIR coefficients
               std::vector<int16> m_taps   = std::vector<int16>{
                        -1, -3, 3, -1, -3, 6, -1, -7, 9, -1, -12, 14, 1, -20, 19, 5,
                        -31, 26, 12, -45, 32, 23, -63, 37, 40, -86, 40, 64, -113, 39, 96, -145,
                        33, 139, -180, 17, 195, -218, -9, 266, -258, -53, 357, -299, -118, 472, -339, -215,
                        620, -376, -360, 822, -409, -585, 1118, -437, -973, 1625, -458, -1801, 2810, -470, -5012, 10783,
                        25067};
               //FIR Graph class
               xf::dsp::aie::fir::sr_sym::fir_sr_sym_graph<cint16, int16,
               FIR129_LENGTH, FIR129_SHIFT, FIR129_ROUND_MODE, FIR129_INPUT_SAMPLES>
               firGraph;
               public:
               port<input> in;
               port<output> out;
               // Constructor - with FIR graph class initialization
               test_kernel():firGraph(m_taps) {
                  // Make connections
                  // Size of window in Bytes.
                  // Margin gets automatically added within the FIR graph class.
                  // Margin equals to FIR length rounded up to nearest multiple of 32
                  Bytes.
                  connect<>(in, firGraph.in);
                  connect<>(firGraph.out, out);
               };
            };
        };

test.cpp
~~~~~~~~
.. code-block::

        #include "test.h"

        simulation::platform<1,1> platform("data/input.txt", "data/output.txt");
        testcase::test_kernel filter ;

        connect<> net0(platform.src[0], filter.in);
        connect<> net1(filter.out, platform.sink[0]);

        int main(void) {
            filter.init() ;
            filter.run() ;
            filter.end() ;
            return 0 ;
        }


.. |image1| image:: ./media/image1.png
.. |image2| image:: ./media/image2.png
.. |image3| image:: ./media/image4.png
.. |image4| image:: ./media/image2.png
.. |image5| image:: ./media/image2.png
.. |image6| image:: ./media/image2.png
.. |image7| image:: ./media/image5.png
.. |image8| image:: ./media/image6.png
.. |image9| image:: ./media/image7.png
.. |image10| image:: ./media/image2.png
.. |image11| image:: ./media/image2.png
.. |image12| image:: ./media/image2.png
.. |image13| image:: ./media/image2.png

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:

