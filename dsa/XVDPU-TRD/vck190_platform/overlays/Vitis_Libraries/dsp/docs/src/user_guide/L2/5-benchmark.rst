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
   
.. _5_BENCHMARK:

=============
Benchmark/QoR
=============

This section provides the L2 performance benchmarks and QoR (Quality of Results) for AIE DSP library elements with various configurations. The results are extracted from hardware emulation based simulations using the Makefile flow defined in: :ref:`2_COMPILING_AND_SIMULATING`.

The QoR are reflected using the below metrics:

- cycleCountAvg         - average cycle count that takes to execute kernel function (not including kernel/window buffer overheads).
- throughputAvg         - input throughput calculated based on `cycleCountAvg`, taking into account input window size.
- initiationInterval    - time that must pass between two consecutive iterations execution starts of a given function, including overheads i.e., time between a function start and its previous start.
- throughpuInitIntAvg   - input throughput calculated based on `initiationInterval`, taking into account input window size.
- NUM_BANKS             - number of memory banks used by the design
- NUM_AIE               - number of AIE tiles used by the design
- DATA_MEMORY           - total data memory in Bytes used by the design
- PROGRAM_MEMORY        - total program memory in Bytes used by the design

In addition, QoR metrics are harvested for each kernel the design consists of. For example a FIR configured to be implemented on two tiles (CASC_LEN=2) will have two sets of figures displayed in the table below (space deliminated).

Filters
~~~~~~~

Following table gives results for FIR filter with a wide variety of supported parameters, which are defined in: :ref:`2_CONFIGURATION_PARAMETERS_FILTERS`

:download:`fir_benchmark.csv <../../csv_data_files/L2/fir_benchmark.csv>`

.. csv-table:: FIR benchmark
   :file: ../../csv_data_files/L2/fir_benchmark.csv
   :align: center
   :header-rows: 1



FFT
~~~

Following table gives results for the FFT/IFFT function with a wide variety of supported parameters, which are defined in: :ref:`2_CONFIGURATION_PARAMETERS_FFT`.

:download:`fft_benchmark.csv <../../csv_data_files/L2/fft_benchmark.csv>`

.. csv-table:: FFT benchmark
   :file: ../../csv_data_files/L2/fft_benchmark.csv
   :align: center
   :header-rows: 1



Matrix Multiply
~~~~~~~~~~~~~~~

Following table gives results for the matrix multiply function with a wide variety of supported parameters, which are defined in: :ref:`2_CONFIGURATION_PARAMETERS_GEMM`.

.. note:: cycleCountAvg does not include the cycle count information for the additional shuffling/tiling widget kernels, but initiationInterval and PROGRAM_MEMORY do include shuffling/tiling widget kernels. 

:download:`mat_mult_benchmark.csv <../../csv_data_files/L2/mat_mult_benchmark.csv>`

.. csv-table:: Matrix Multiply benchmark
   :file: ../../csv_data_files/L2/mat_mult_benchmark.csv
   :align: center
   :header-rows: 1



Widgets
~~~~~~~

Following table gives results for the widgets with a wide variety of supported parameters, which are defined in: :ref:`2_CONFIGURATION_PARAMETERS_WIDGETS`.

:download:`widget_benchmark.csv <../../csv_data_files/L2/widget_benchmark.csv>`

.. csv-table:: Widgets benchmark
   :file: ../../csv_data_files/L2/widget_benchmark.csv
   :align: center
   :header-rows: 1




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


