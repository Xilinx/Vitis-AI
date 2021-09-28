.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

..
   Copyright 2019 Xilinx, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


*****************
GeoIP Engine
*****************

Overview
========

GeoIP Engine is an engine of Query geographic location ID based on IP. It only supports ipv4, and ip can be a number or a string ("5.157.15.64" <-> 94179136), and output geo info id. According to the id, the required information can be found in GeoIP2/GeoLite2 (csv format) from Maxmind.

Implementation
================

The following is an introduction from two parts: input requirements and kernel design

Input requirements
==================

The data source of the input data comes from networks field in GeoIP2/GeoLite2, such as "5.157.15.0/24". It can be represented by 37bit. It is divided into two parts here: `netHigh16` and `netLow21`.

- `netHigh16`: Its low 32bit stores its starting line in the file, and its high 32bit stores its starting position in `netLow21`.
- `netLow21`: Its storage is divided into two situations. When the number of `netLow21` corresponding to one `netHigh16` is less than the threshold, it is stored in a 512bit array tightly. If it is greater than the threshold, the index is stored first, and then the content.

For more details, please refer to the source code.

Kernel Design
=============

The GeoIP Kernel design is show in the figure below:

.. _my-figure1:
.. figure:: /images/text/geoip_kernel.png
    :alt: Figure 1 GeoIP Kenrel architecture on FPGA
    :width: 80%
    :align: center

The functions of each module in the figure are:

- `findAddrID`: find start address in `netLow21` according to input ip
- `readHead` and `findChunck`: Narrow the search scope in `netLow21`
- `addrConvert`, `readNetLow21` and `searchID`: Find the `netLow21` corresponding to ip and get its id.

Resource Utilization
====================
 
The GeoIP is validated on Alveo U200 card.
The hardware resources utilization are listed in the table above (not include Platform).

+------------------+---------+---------+--------+--------+-------+
|       Name       |   LUT   |   REG   |  BRAM  | URAM   | DSP   |
+------------------+---------+---------+--------+--------+-------+
|                  | 13,241  | 23,732  |   23   |  16    |  8    |
|   GeoIP_kernel   +---------+---------+--------+--------+-------+
|                  |  0.84%  |  0.28%  | 0.96%  | 1.25%  | 0.07% |
+------------------+---------+---------+--------+--------+-------+

.. toctree::
   :maxdepth: 1
